"""
generate.py

Inference helpers for GPT generation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
if str(MODEL_DIR) not in sys.path:
	sys.path.insert(0, str(MODEL_DIR))


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, map_location: str = "cpu") -> dict:
	"""Load model weights from a saved checkpoint payload."""
	payload = torch.load(checkpoint_path, map_location=map_location)
	model.load_state_dict(payload["model_state_dict"])
	return payload


def decode_token_ids(token_ids: list[int], tokenizer) -> str:
	"""
	Decode token ids using tokenizer.id_to_token mapping.

	This is an approximate decode for subword tokens.
	"""
	id_to_token = tokenizer.get_id_to_token()
	pieces = []
	for idx in token_ids:
		tok = id_to_token.get(int(idx), "<UNK>")
		tok = tok.replace("</w>", "")
		pieces.append(tok)

	text = "".join(pieces)
	return text.replace("▁", " ").strip()


def generate_from_prompt(
	model: torch.nn.Module,
	tokenizer,
	prompt: str,
	max_new_tokens: int = 40,
	temperature: float = 1.0,
	top_k: int | None = 20,
	device: str = "cpu",
) -> tuple[list[int], str]:
	"""
	Encode prompt, generate continuation, then decode.
	"""
	model.eval()
	prompt_ids = tokenizer.encode(prompt)
	if not prompt_ids:
		prompt_ids = [tokenizer.get_token_to_id().get("<BOS>", 0)]

	input_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
	max_length = len(prompt_ids) + max_new_tokens

	with torch.no_grad():
		out_tokens = model.generate(
			initial_tokens=input_tokens,
			max_length=max_length,
			temperature=temperature,
			top_k=top_k,
		)

	token_ids = out_tokens[0].detach().cpu().tolist()
	text = decode_token_ids(token_ids, tokenizer)
	return token_ids, text
