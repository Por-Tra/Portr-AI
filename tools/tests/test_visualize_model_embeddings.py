"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : test_visualize_model_embeddings.py

Numerical and artifact checks for the model embedding explorer.

"""
import json
import hashlib
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import torch

from model.gpt_model import GPT
from tokenizer.tokenizer import Tokenizer
from tools.visualize_model_embeddings import (contextual_vectors, cosine_neighbors,
    load_weights, model_from_snapshot, parse_args, project, safe_json, sample_indices, write_viewer)


class ModelEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_neighbors_are_original_cosines_excluding_self_and_zero_vectors(self):
        vectors = np.array([[1, 0], [100, .1], [-1, 0], [0, 1], [0, 0]], dtype=np.float32)
        neighbors, scores = cosine_neighbors(vectors, np.array([0, 2, 4]), 4)
        self.assertEqual(neighbors[0][0], 1)  # Scaling the norm must not change cosine proximity.
        self.assertGreater(scores[0][0], .999)
        self.assertNotIn(0, neighbors[0])
        self.assertNotIn(4, neighbors[0])
        self.assertEqual(neighbors[2], [])
        for row, values, ids in zip([0, 2], scores, neighbors):
            for value, other in zip(values, ids):
                expected = vectors[row] @ vectors[other] / (np.linalg.norm(vectors[row]) * np.linalg.norm(vectors[other]))
                self.assertAlmostEqual(value, float(expected), places=6)

    def test_contextual_vectors_match_real_forward_and_preserve_causality(self):
        torch.manual_seed(42)
        model = GPT(16, 8, embedding_dim=8, num_heads=2, num_blocks=1, dropout=.4).eval()
        class TinyTokenizer:
            id_to_token = {1: 'a', 2: 'b', 3: 'c'}
            def encode(self, text):
                return [{'a':1, 'b':2, 'c':3}[word] for word in text.split()]
        tokenizer = TinyTokenizer()
        captured = []
        handle = model.final_ln.register_forward_hook(lambda module, inputs, output: captured.append(output.detach()))
        with torch.inference_mode():
            model(torch.tensor([[1, 2, 1]]))
        handle.remove()
        vectors, rows = contextual_vectors(model, tokenizer, ['a b a', 'a b c'], 'cpu')
        np.testing.assert_allclose(vectors[:3], captured[0][0].numpy(), atol=1e-6)
        np.testing.assert_allclose(vectors[:2], vectors[3:5], atol=1e-6)
        self.assertFalse(np.allclose(vectors[0], vectors[2]))
        self.assertEqual([r['token_id'] for r in rows], [1,2,1,1,2,3])
        self.assertEqual(rows[3]['position'], 0)
        self.assertEqual(rows[3]['text_number'], 2)
        with self.assertRaisesRegex(ValueError, 'contexte'):
            contextual_vectors(model, tokenizer, [' '.join(['a'] * 9)], 'cpu')

    def test_projection_and_sampling_are_reproducible_and_validate_input(self):
        x = np.random.default_rng(4).normal(size=(30, 8)).astype(np.float32)
        a, info = project(x, 'pca', 3, 7, True)
        b, _ = project(x, 'pca', 3, 7, True)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (30, 3))
        self.assertGreater(sum(info['explained_variance_ratio']), 0)
        np.testing.assert_array_equal(sample_indices(100, 10, 7), sample_indices(100, 10, 7))
        self.assertGreater(sample_indices(100, 10, 7).max(), 9)
        with self.assertRaises(ValueError):
            project(x[:2], 'pca', 3, 7, True)
        with self.assertRaises(ValueError):
            project(np.ones((4, 8)), 'pca', 2, 7, True)
        with self.assertRaises(ValueError):
            cosine_neighbors(np.array([[1, float('nan')], [0, 1]]), np.array([0]), 1)

    def test_tokenizer_mismatch_is_rejected_before_plotting(self):
        tokenizer = Tokenizer()
        tokenizer.token_to_id = {t: i for i, t in enumerate(sorted(tokenizer.vocab))}
        tokenizer.id_to_token = {i:t for t,i in tokenizer.token_to_id.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'model_state_dict': {'embedding.token_emb.weight': torch.randn(len(tokenizer.vocab), 8)},
                        'tokenizer_fingerprint': 'wrong-tokenizer'}, path)
            with self.assertRaisesRegex(ValueError, 'Tokenizer incompatible'):
                load_weights(path, tokenizer)

    def test_snapshot_digest_and_context_model_survive_source_replacement(self):
        tokenizer = Tokenizer()
        tokenizer.token_to_id = {t: i for i, t in enumerate(sorted(tokenizer.vocab))}
        tokenizer.id_to_token = {i:t for t,i in tokenizer.token_to_id.items()}
        hparams = dict(vocab_size=len(tokenizer.vocab), seq_length=8, embedding_dim=8,
                       num_heads=2, num_blocks=1, dropout=0)
        model = GPT(**hparams).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'model_state_dict': model.state_dict(), 'model_hparams': hparams,
                        'tokenizer_fingerprint': tokenizer.fingerprint()}, path)
            expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            payload, state, digest = load_weights(path, tokenizer)
            path.write_bytes(b'A later training snapshot replaced this file')
            self.assertEqual(digest, expected_digest)
            restored = model_from_snapshot(payload, 'cpu')
            with torch.inference_mode():
                x = torch.tensor([[0, 1, 2]])
                torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)

    def test_offline_html_escapes_tokens_and_retains_neighbors_and_ids(self):
        malicious = '</script><script>alert("token")</script>{{MODEL}}'
        encoded = safe_json({'token': malicious})
        self.assertNotIn('</script>', encoded)
        self.assertEqual(json.loads(encoded)['token'], malicious)
        rows = [{'id': i, 'token': token, 'group': 'Lettres', 'norm': 1.0}
                for i, token in enumerate([malicious, 'chat', 'chien', 'bleu'])]
        coordinates = np.arange(8, dtype=np.float32).reshape(4,2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'view.html'
            write_viewer(path, rows, np.arange(4), coordinates, [[1],[0],[1],[2]], [[.5]]*4,
                {'kind':'tokens','method':'pca','model_label':'SFT','features':8,'context':16,
                 'checkpoint':'model.pt','normalize':True})
            page = path.read_text(encoding='utf-8')
            match = re.search(r'<script id="embedding-data" type="application/json">(.*?)</script>', page, re.S)
            self.assertEqual(json.loads(match[1])['rows'][0]['token'], malicious)
            self.assertNotIn('<script src=', page)
            self.assertIn('plotly.js', page)
            self.assertEqual(json.loads(match[1])['neighbors'][0], [1])

    def test_cli_defaults_to_active_sft_and_accepts_checkpoint_alias(self):
        self.assertEqual(parse_args([]).model, 'sft')
        self.assertEqual(parse_args(['--checkpoint', 'custom.pt']).model, 'custom.pt')


if __name__ == '__main__':
    unittest.main()
