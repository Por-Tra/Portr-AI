"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : test_sft_best_model.py

The inference model must stay at the best validation step, not the last.

"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from model.gpt_model import GPT
from training.sft_train import SFTConfig, train_sft_model
from training.train import TrainConfig
import training.train as training


class BestModelTests(unittest.TestCase):
    def test_sft_keeps_best_weights_and_last_optimizer_checkpoint_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.manual_seed(19)
            model = GPT(8, 4, embedding_dim=8, num_heads=2, num_blocks=1, dropout=0)
            x = torch.randint(0, 8, (4, 4))
            data = DataLoader(TensorDataset(x, x.roll(-1, 1)), batch_size=2)
            config = SFTConfig(epochs=2, learning_rate=1e-3, device="cpu", use_amp=False,
                warmup_steps=0, eval_interval=0, checkpoint_every_steps=0,
                checkpoint_dir=str(root/"checkpoints"), final_model_dir=str(root/"models"),
                benchmark_path=str(root/"metrics.csv"))
            captured = []
            original_save = training.save_checkpoint
            def capture(**kwargs):
                original_save(**kwargs)
                training.wait_for_pending_checkpoint()
                if kwargs["epoch"] == 1:
                    captured.append(torch.load(kwargs["checkpoint_path"], weights_only=True))
            with patch("training.train.evaluate", side_effect=[1., 2.]), \
                 patch("training.train.save_checkpoint", side_effect=capture):
                history = train_sft_model(model, data, config, val_dataloader=data)
            best = torch.load(history["final_model_path"], weights_only=True)
            last = torch.load(history["checkpoint_path"], weights_only=True)
            self.assertEqual(best["selection"]["epoch"], 1)
            self.assertEqual(best["selection"]["val_loss"], 1.)
            self.assertEqual(history["best_epoch"], 1)
            self.assertEqual(last["epoch"], 2)
            self.assertEqual(last["step"], 4)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, best["model_state_dict"][key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, last["model_state_dict"][key])
                                for key,value in best["model_state_dict"].items()))
            self.assertFalse(TrainConfig().save_best_model_state)
            self.assertTrue(SFTConfig().save_best_model_state)

            # Resume the actual epoch-one checkpoint, preserving an earlier best.
            torch.save(captured[0], history["checkpoint_path"])
            resumed = GPT(8, 4, embedding_dim=8, num_heads=2, num_blocks=1, dropout=0)
            with patch("training.train.evaluate", return_value=2.):
                resumed_history = train_sft_model(resumed, data, config, val_dataloader=data)
            self.assertEqual(resumed_history["best_epoch"], 1)
            for key,value in resumed.state_dict().items():
                torch.testing.assert_close(value, best["model_state_dict"][key], rtol=0, atol=0)

            torch.save(captured[0], history["checkpoint_path"])
            Path(history["final_model_path"]).unlink()
            with self.assertRaisesRegex(RuntimeError, "Meilleur modele de validation absent"):
                train_sft_model(resumed, data, config, val_dataloader=data)


if __name__ == "__main__":
    unittest.main()
