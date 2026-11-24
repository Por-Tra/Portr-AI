"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : test_diagnose_sft.py

Checks for the controls and statistics used in the SFT investigation.
"""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.diagnose_sft import audit_dataset, bootstrap_delta, shuffled_prompts, summary, write_json


class TokenizerStub:
    def get_token_to_id(self):
        return {"<PAD>": 0, "<|user|>": 1, "<|assistant|>": 2, "<|end|>": 3}

    def get_id_to_token(self):
        return {i: str(i) for i in range(100)}


def dataset(pairs, length=20):
    xs, ys = [], []
    for prompt, answer in pairs:
        ids = [1, *prompt, 2, *answer, 3]
        x = np.full(length, 0, dtype="int32")
        y = np.full(length, -100, dtype="int32")
        x[:len(ids)] = ids
        y[len(prompt)+1:len(ids)-1] = [*answer, 3]
        xs.append(x)
        ys.append(y)
    class Data(SimpleNamespace):
        def __len__(self):
            return len(self.labels)
    return Data(input_ids=np.array(xs), labels=np.array(ys))


class DiagnosticTests(unittest.TestCase):
    def test_shuffled_control_preserves_answers_alignment_and_no_self_donors(self):
        val = dataset([([10], [40, 41]), ([20, 21, 22], [50]), ([30, 31], [60, 61, 62])])
        changed = shuffled_prompts(val, TokenizerStub())
        for i, (x, y) in enumerate(changed):
            x, y = x.numpy(), y.numpy()
            original = val.labels[i]
            np.testing.assert_array_equal(y[y != -100], original[original != -100])
            mask = y[:-1] != -100
            np.testing.assert_array_equal(y[:-1][mask], x[1:][mask])
            self.assertEqual(y[-1], -100)
            self.assertNotEqual(int(x[1]), int(val.input_ids[i, 1]))
            self.assertEqual(x[np.flatnonzero(y != -100)[0]], 2)

    def test_long_donor_is_cropped_without_truncating_the_answer(self):
        val = dataset([([10]*12, [40]), ([20], [50]*12)], length=18)
        changed = shuffled_prompts(val, TokenizerStub())
        for i, (x, y) in enumerate(changed):
            np.testing.assert_array_equal(y.numpy()[y.numpy() != -100], val.labels[i][val.labels[i] != -100])
            self.assertEqual(len(x), 18)
            self.assertEqual(y[-1].item(), -100)

    def test_audit_distinguishes_prompt_overlap_from_exact_answer_leakage(self):
        train = dataset([([10], [40]), ([10], [40]), ([20], [50])])
        val = dataset([([10], [41]), ([30], [60])])
        result = audit_dataset(train, val, TokenizerStub())
        self.assertEqual(result["train"]["duplicate_pairs"], 1)
        self.assertEqual(result["validation_pair_overlap"], 0)
        self.assertEqual(result["validation_prompt_overlap_indices"], [0])

    def test_paired_bootstrap_uses_token_weighting_and_direction(self):
        counts = np.array([2, 12, 50])
        b = np.array([3., 20., 150.])
        a = b - .25 * counts
        result = bootstrap_delta(a, b, counts)
        self.assertAlmostEqual(result["delta_ce"], -.25)
        np.testing.assert_allclose(result["ci95"], [-.25, -.25])

    def test_summary_persists_seen_and_unseen_groups_with_their_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            dest = out / "measurements" / "base"
            dest.mkdir(parents=True)
            write_json(out / "data_audit.json", {"validation_prompt_overlap_indices": [0]})
            write_json(dest / "metrics.json", {
                "sft_validation": {"ce": 2., "end_accuracy": .5},
                "sft_train_sample": {"ce": 1.9}, "corpus_validation_full": {"ce": 3.},
                "corpus_train_sample": {"ce": 2.9}, "shuffled_prompt_delta": {"delta_ce": .3}})
            write_json(dest / "generations.json", [])
            np.savez(dest / "per_example.npz", sft_validation_sums=np.array([4., 36., 100.]),
                     sft_validation_counts=np.array([2, 12, 50]))
            with patch("tools.diagnose_sft.plot_summary"), contextlib.redirect_stdout(io.StringIO()):
                summary(out)
            result = json.loads((out / "comparison.json").read_text())["paired_comparisons"]["base"]
            self.assertEqual(result["seen_prompts"], {"examples": 1, "ce": 2.})
            self.assertEqual(result["unseen_prompts"]["examples"], 2)
            self.assertEqual(result["vs_base_unseen_prompts"]["delta_ce"], 0.)


if __name__ == "__main__":
    unittest.main()
