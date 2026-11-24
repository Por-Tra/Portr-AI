"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : test_sft_preparation.py

Leakage and short-answer regressions for SFT data preparation."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_sft_data as prep
import main
from data.sft_dataset import SFTDataset
from tokenizer.tokenizer import Tokenizer
from tools.repair_sft import curriculum


def example(prompt, answer):
    return prep.build_example(prompt, answer, lambda text: [ord(c) for c in text],
                              (1, 2, 3, 0), 32, 1, 20, None)[0]


class SFTPreparationTests(unittest.TestCase):
    def test_explicit_holdouts_survive_repreparation_and_reject_leakage(self):
        rows = [example("Q", "A"), example("R", "B"), example("S", "C")]
        train, val, stats = prep.split_prompt_groups(rows, .9, 71,
            ["a", "a", "b"], split_ids=["train", "train", "dev"])
        self.assertEqual(train, rows[:2])
        self.assertEqual(val, rows[2:])
        self.assertTrue(stats["explicit_splits"])
        with self.assertRaisesRegex(ValueError, "traverse"):
            prep.split_prompt_groups(rows, .1, 42, ["a", "a", "b"], split_ids=["train", "val", "val"])
        with self.assertRaisesRegex(ValueError, "tous"):
            prep.split_prompt_groups(rows, .1, 42, split_ids=["train", None, "val"])
        with self.assertRaisesRegex(ValueError, "tous"):
            prep.split_prompt_groups(rows, .1, 42, split_ids=["train", "test", "val"])

    def test_one_token_answer_and_its_end_are_supervised(self):
        x, y = example("Q", "4")
        self.assertEqual([v for v in y if v != -100], [ord("4"), 3])
        self.assertEqual(x[y.index(ord("4"))], 2)

    def test_group_split_never_shares_a_question_with_another_answer(self):
        examples = [example("Q", "A"), example("Q", "B"), example("R", "C"), example("S", "D")]
        train, val, stats = prep.split_prompt_groups(examples, .3, 42)
        self.assertFalse({prep.prompt_key(e) for e in train} & {prep.prompt_key(e) for e in val})
        self.assertTrue(train and val)
        self.assertEqual(stats["prompt_overlap"], 0)

    def test_pairs_deduplicate_and_paraphrases_stay_in_one_declared_group(self):
        examples = [example("Q", "A"), example("Q", "A"), example("R", "A"), example("S", "B")]
        train, val, stats = prep.split_prompt_groups(examples, .3, 42, ["task", "task", "task", "other"])
        self.assertEqual(stats["duplicates_removed"], 1)
        q_in_train = any(prep.prompt_key(e) == prep.prompt_key(examples[0]) for e in train)
        r_in_train = any(prep.prompt_key(e) == prep.prompt_key(examples[2]) for e in train)
        self.assertEqual(q_in_train, r_in_train)
        self.assertEqual(len(train) + len(val), 3)

    def test_impossible_split_and_conflicting_groups_fail(self):
        with self.assertRaisesRegex(ValueError, "deux groupes"):
            prep.split_prompt_groups([example("Q", "A"), example("Q", "B")], .1, 42)
        with self.assertRaisesRegex(ValueError, "plusieurs groupes"):
            prep.split_prompt_groups([example("Q", "A"), example("Q", "B")], .1, 42, ["one", "two"])

    def test_identity_question_is_reserved_to_training(self):
        e = [example("Q", "A"), example("R", "B"), example("S", "C")]
        train, val, _ = prep.split_prompt_groups(e, .8, 42, reserved_prompts=[prep.prompt_key(e[0])])
        self.assertIn(prep.prompt_key(e[0]), {prep.prompt_key(x) for x in train})
        self.assertNotIn(prep.prompt_key(e[0]), {prep.prompt_key(x) for x in val})

    def test_curriculum_is_deterministic_and_test_groups_are_held_out(self):
        pool, test = curriculum()
        self.assertEqual((pool, test), curriculum())
        a = {r["group"] for r in pool if r["family"] != "faits_appris"}
        b = {r["group"] for r in test if r["family"] != "faits_appris"}
        self.assertFalse(a & b)
        self.assertFalse({r["prompt"] for r in pool} & {r["prompt"] for r in test})
        for row in pool + test:
            if row["family"] in ("addition", "soustraction", "comparaison"):
                import ast
                left, right = ast.literal_eval(row["group"].split(":", 1)[1])
                expected = {"addition": left+right, "soustraction": left-right, "comparaison": max(left,right)}[row["family"]]
                self.assertEqual(row["answer"], str(expected))

    def test_local_preparation_and_manager_accept_only_grouped_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tok = Tokenizer()
            tok.vocab.update("_QRST4BCD")
            tok.token_to_id = {word:i for i,word in enumerate(sorted(tok.vocab))}
            tok.id_to_token = {i:word for word,i in tok.token_to_id.items()}
            tok.save(root/"tokenizer.json")
            source = root/"source.jsonl"
            source.write_text("\n".join(json.dumps({"prompt":p,"answer":a})
                               for p,a in zip("QRST","4BCD")),encoding="utf-8")
            args = argparse.Namespace(seq_length=16,val_ratio=.25,max_examples=0,split="local",list_sources=False,
                tokenizer_path=root/"tokenizer.json",no_update_tokenizer=True,seed=42,context_mode="merge",
                min_answer_tokens=1,max_answer_tokens=10,max_prompt_tokens=None,identity_jsonl=None,
                out_prefix=root/"sft",local_jsonl=source)
            with patch.object(prep,"load_dataset",side_effect=AssertionError("No network")):
                prep.prepare(args)
            train,val = SFTDataset(root/"sft_train.npz"),SFTDataset(root/"sft_val.npz")
            self.assertEqual((len(train),len(val)),(3,1))
            with patch.multiple(main,SFT_TRAIN_PATH=root/"sft_train.npz",SFT_VAL_PATH=root/"sft_val.npz",
                                SFT_META_PATH=root/"sft_meta.json",TOKENIZER_MODEL_PATH=str(root/"tokenizer.json"),SEQ_LENGTH=16,SFT_SEQ_LENGTH=16):
                self.assertTrue(main._sft_data_ready())
                meta=json.loads((root/"sft_meta.json").read_text())
                meta["split_groups"]=None
                (root/"sft_meta.json").write_text(json.dumps(meta),encoding="utf-8")
                self.assertFalse(main._sft_data_ready())


if __name__ == "__main__":
    unittest.main()
