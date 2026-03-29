"""
optimizer.py

Optimizer utilities for GPT training.
"""

from __future__ import annotations

import torch


def create_adamw_optimizer(model: torch.nn.Module, lr: float = 3e-4, weight_decay: float = 0.01, betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.Optimizer:
	"""
	Build AdamW optimizer with decoupled weight decay.

	We avoid weight decay on biases and norm parameters.
	"""
	decay_params = []
	no_decay_params = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue

		if name.endswith("bias") or "norm" in name.lower() or "ln" in name.lower():
			no_decay_params.append(param)
		else:
			decay_params.append(param)

	param_groups = [
		{"params": decay_params, "weight_decay": weight_decay},
		{"params": no_decay_params, "weight_decay": 0.0},
	]

	return torch.optim.AdamW(param_groups, lr=lr, betas=betas)

