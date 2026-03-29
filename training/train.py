"""
train.py

Training loop for GPT model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import torch


@dataclass
class TrainConfig:
	epochs: int = 1
	device: str = "auto"
	grad_clip: float = 1.0
	log_interval: int = 20
	max_steps_per_epoch: int | None = None
	checkpoint_dir: str = "artifacts/checkpoints"
	checkpoint_name: str = "gpt_latest.pt"


def resolve_device(requested: str = "auto") -> torch.device:
	if requested == "auto":
		if torch.cuda.is_available():
			return torch.device("cuda")
		return torch.device("cpu")
	return torch.device(requested)


def save_checkpoint(
	model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int, avg_loss: float, checkpoint_path: Path,) -> None:
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"epoch": epoch,
		"step": step,
		"avg_loss": avg_loss,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
	}
	torch.save(payload, checkpoint_path)


def train_model(model: torch.nn.Module, dataloader, optimizer: torch.optim.Optimizer,loss_fn, config: TrainConfig,
) -> dict:
	"""
	Train a GPT-like model and return metrics/checkpoint metadata.
	"""
	device = resolve_device(config.device)
	model.to(device)
	model.train()

	ckpt_path = Path(config.checkpoint_dir) / config.checkpoint_name

	history = {
		"epoch_losses": [],
		"checkpoint_path": str(ckpt_path),
		"device": str(device),
	}

	global_step = 0

	for epoch in range(config.epochs):
		running_loss = 0.0
		steps = 0
		start_t = time.time()

		for batch_idx, (x, y) in enumerate(dataloader):
			if config.max_steps_per_epoch is not None and batch_idx >= config.max_steps_per_epoch:
				break

			x = x.to(device)
			y = y.to(device)

			optimizer.zero_grad(set_to_none=True)
			logits = model(x)
			loss = loss_fn(logits, y)
			loss.backward()

			if config.grad_clip is not None and config.grad_clip > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

			optimizer.step()

			loss_item = float(loss.item())
			running_loss += loss_item
			steps += 1
			global_step += 1

			if (batch_idx + 1) % config.log_interval == 0:
				print(
					f"[train] epoch={epoch + 1}/{config.epochs} "
					f"step={batch_idx + 1} loss={loss_item:.4f}"
				)

		avg_loss = running_loss / max(steps, 1)
		elapsed = time.time() - start_t
		history["epoch_losses"].append(avg_loss)

		print(
			f"[train] epoch {epoch + 1} done | "
			f"avg_loss={avg_loss:.4f} | steps={steps} | time={elapsed:.1f}s"
		)

		save_checkpoint(
			model=model,
			optimizer=optimizer,
			epoch=epoch + 1,
			step=global_step,
			avg_loss=avg_loss,
			checkpoint_path=ckpt_path,
		)
		print(f"[train] checkpoint saved: {ckpt_path}")

	return history

