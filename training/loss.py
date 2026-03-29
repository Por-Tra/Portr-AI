"""
loss.py

Loss functions for GPT-style autoregressive training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def autoregressive_cross_entropy(
	logits: torch.Tensor,
	targets: torch.Tensor,
	ignore_index: int = -100,
) -> torch.Tensor:
	"""
	Compute token-level cross entropy for language modeling.

	Args:
		logits: [batch, seq, vocab_size]
		targets: [batch, seq]
		ignore_index: target id to ignore in loss computation

	Returns:
		Scalar loss tensor.
	"""
	if logits.dim() != 3:
		raise ValueError(f"Expected logits dim=3, got shape={tuple(logits.shape)}")
	if targets.dim() != 2:
		raise ValueError(f"Expected targets dim=2, got shape={tuple(targets.shape)}")

	bsz, seq_len, vocab_size = logits.shape
	if targets.shape != (bsz, seq_len):
		raise ValueError(
			"targets shape must match [batch, seq] from logits; "
			f"got logits={tuple(logits.shape)}, targets={tuple(targets.shape)}"
		)

	flat_logits = logits.reshape(-1, vocab_size)
	flat_targets = targets.reshape(-1)
	return F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)


def perplexity_from_loss(loss_value: float) -> float:
	"""Compute perplexity from scalar cross-entropy loss."""
	return float(torch.exp(torch.tensor(loss_value)).item())

