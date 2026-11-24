"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : test_sft_inference.py

"""

import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import torch
import main


class SFTInferenceTests(unittest.TestCase):
    def test_sft_and_base_keep_separate_decoding_settings(self):
        for sft in (False, True):
            with patch("builtins.input",side_effect=["question",":quit"]), \
                 patch.object(main,"generate_from_prompt",return_value=([1],"reponse")) as generate, \
                 contextlib.redirect_stdout(io.StringIO()):
                main._interactive_reprompt_loop(Mock(),Mock(),torch.device("cpu"),sft)
            expected = main.SFT_INFER_TEMPERATURE if sft else main.INFER_TEMPERATURE
            self.assertEqual(generate.call_args.kwargs["temperature"],expected)
            self.assertEqual(generate.call_args.kwargs["use_sft_template"],sft)

    def test_sft_menu_starts_with_the_users_question(self):
        with patch.object(main,"_assert_no_incomplete_preparation"), \
             patch.object(main.Tokenizer,"load",return_value=Mock()), \
             patch.object(main,"_sft_model_path",return_value=Path("sft_model.pt")), \
             patch.object(main,"_load_model_for_inference_from_artifact",return_value=Mock()), \
             patch.object(main,"_run_inference") as canned, \
             patch.object(main,"_interactive_reprompt_loop") as interactive:
            main._inference_action(torch.device("cpu"),sft=True)
        canned.assert_not_called()
        self.assertTrue(interactive.call_args.kwargs["use_sft_template"])
