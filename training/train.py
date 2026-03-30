"""
train.py — version optimisée
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
import math

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

    # ── Scheduler ────────────────────────────────────────────────────
    warmup_steps: int = 100          # steps de montée progressive du LR
    lr_decay_factor: float = 0.1     # LR final = LR_initial × lr_decay_factor

    # ── Mixed precision ──────────────────────────────────────────────
    use_amp: bool = True             # float16 sur GPU (ignoré sur CPU)

    # ── Gradient accumulation ────────────────────────────────────────
    grad_accum_steps: int = 1        # accumule N batches avant optimizer.step()


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_factor: float,
) -> float:
    """
    Retourne le multiplicateur du LR pour le step courant.

    Phase 1 (warmup) : montée linéaire de 0 → 1
    Phase 2 (decay)  : descente cosine de 1 → min_lr_factor
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_factor + (1.0 - min_lr_factor) * cosine


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    avg_loss: float,
    checkpoint_path: Path,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "avg_loss": avg_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    torch.save(payload, checkpoint_path)


def train_model(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    config: TrainConfig,
) -> dict:
    device = resolve_device(config.device)
    model.to(device)
    model.train()

    # ── Total steps pour le scheduler ────────────────────────────────
    steps_per_epoch = (
        min(len(dataloader), config.max_steps_per_epoch)
        if config.max_steps_per_epoch else len(dataloader)
    )
    total_steps = steps_per_epoch * config.epochs

    # ── 1. Scheduler cosine + warmup ─────────────────────────────────
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_lr_with_warmup(
            step,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr_factor=config.lr_decay_factor,
        ),
    )

    # ── 2. Mixed precision (AMP) — actif seulement sur GPU ───────────
    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(dataloader):
            if config.max_steps_per_epoch is not None and batch_idx >= config.max_steps_per_epoch:
                break

            x = x.to(device)
            y = y.to(device)

            # ── 3. Gradient accumulation : on divise la loss ──────────
            # pour que les gradients accumulés soient équivalents à un
            # seul batch de taille batch_size × grad_accum_steps.
            is_last_accum = (batch_idx + 1) % config.grad_accum_steps == 0

            # ── Forward en mixed precision ────────────────────────────
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y) / config.grad_accum_steps

            scaler.scale(loss).backward()

            if is_last_accum:
                # ── Grad clip + optimizer step ────────────────────────
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_clip
                    )
                else:
                    grad_norm = 0.0

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()          # ← après chaque optimizer step
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            # loss réelle (× grad_accum pour annuler la division)
            loss_item = float(loss.item()) * config.grad_accum_steps
            running_loss += loss_item
            steps += 1

            if (batch_idx + 1) % config.log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[train] epoch={epoch + 1}/{config.epochs} "
                    f"step={batch_idx + 1} "
                    f"loss={loss_item:.4f} "
                    f"lr={current_lr:.2e} "
                    f"grad_norm={float(grad_norm):.3f}"
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
            scheduler=scheduler,
            epoch=epoch + 1,
            step=global_step,
            avg_loss=avg_loss,
            checkpoint_path=ckpt_path,
        )
        print(f"[train] checkpoint saved: {ckpt_path}")

    return history