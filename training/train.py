"""
train.py — version optimisée
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import math

import torch


@dataclass
class TrainConfig:
    """classe utiliser pour stocker de l'information
    """
    epochs: int = 1
    device: str = "auto"
    grad_clip: float = 1.0
    log_interval: int = 20
    max_steps_per_epoch: int | None = None
    checkpoint_dir: str = "artifacts/checkpoints"
    checkpoint_name: str = "gpt_latest.pt"
    resume_from_checkpoint: bool = True
    checkpoint_every_steps: int = 200
    save_final_model_state: bool = True
    final_model_dir: str = "artifacts/models"
    final_model_name: str = "gpt_final_state.pt"
    max_cpu_threads: int | None = None
    max_cpu_interop_threads: int | None = None

    #* ── Scheduler ────────────────────────────────────────────────────
    warmup_steps: int = 100          # steps de montée progressive du LR
    lr_decay_factor: float = 0.1     # LR final = LR_initial × lr_decay_factor

    #* ── Mixed precision ──────────────────────────────────────────────
    use_amp: bool = True             # float16 sur GPU (ignoré sur CPU)

    #* ── Gradient accumulation ────────────────────────────────────────
    grad_accum_steps: int = 1        # accumule N batches avant optimizer.step()


def resolve_device(requested: str = "auto") -> torch.device:
    """Resout le device d'execution pour l'entrainement.

    Args:
        requested (str, optional): Device demande ("auto", "cpu", "cuda", ...).
            Defaults to "auto".

    Returns:
        torch.device: Device PyTorch final a utiliser.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _cosine_lr_with_warmup( step: int, warmup_steps: int, total_steps: int, min_lr_factor: float) -> float:
    """Calcule le multiplicateur du learning rate avec warmup + cosine decay.

    Args:
        step (int): Step d'optimisation courant.
        warmup_steps (int): Nombre de steps de warmup lineaire.
        total_steps (int): Nombre total de steps d'optimisation prevus.
        min_lr_factor (float): Facteur minimum du LR en fin de decay.

    Returns:
        float: Multiplicateur a appliquer au LR initial.

    Phase 1 (warmup) : montée linéaire de 0 → 1
    Phase 2 (decay)  : descente cosine de 1 → min_lr_factor
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_factor + (1.0 - min_lr_factor) * cosine


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int,
    step: int, avg_loss: float, checkpoint_path: Path, epoch_losses: list[float] | None = None,
    scaler: torch.amp.GradScaler | None = None) -> None:
    """Sauvegarde l'etat d'entrainement pour reprise ulterieure.

    Args:
        model (torch.nn.Module): Modele entraine.
        optimizer (torch.optim.Optimizer): Optimizer courant.
        scheduler: Scheduler de LR.
        epoch (int): Epoch a sauvegarder.
        step (int): Step global a sauvegarder.
        avg_loss (float): Loss moyenne recente.
        checkpoint_path (Path): Chemin du fichier checkpoint.
        epoch_losses (list[float] | None, optional): Historique des losses par epoch.
        scaler (torch.amp.GradScaler | None, optional): Etat AMP pour reprise fidele.

    Returns:
        None: Cette fonction ecrit le checkpoint sur disque.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "avg_loss": avg_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if epoch_losses is not None:
        payload["epoch_losses"] = list(epoch_losses)
    torch.save(payload, checkpoint_path)


def load_checkpoint_if_available(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, checkpoint_path: Path,
    resume: bool, scaler: torch.amp.GradScaler | None = None) -> tuple[int, int, list[float]]:
    """Charge un checkpoint si disponible et autorise.

    Args:
        model (torch.nn.Module): Modele dans lequel charger les poids.
        optimizer (torch.optim.Optimizer): Optimizer a restaurer.
        scheduler: Scheduler a restaurer.
        checkpoint_path (Path): Chemin du checkpoint.
        resume (bool): Active ou non la reprise.
        scaler (torch.amp.GradScaler | None, optional): Scaler AMP a restaurer si present.

    Returns:
        tuple[int, int, list[float]]: Epoch de depart, step global, historique des losses.
    """
    if not resume or not checkpoint_path.exists():
        return 0, 0, []

    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])

    start_epoch = int(payload.get("epoch", 0))
    global_step = int(payload.get("step", 0))
    epoch_losses = [float(v) for v in payload.get("epoch_losses", [])]

    print(
        f"[train] resume checkpoint: epoch={start_epoch} "
        f"step={global_step} path={checkpoint_path}"
    )
    return start_epoch, global_step, epoch_losses


def save_final_model_state(model: torch.nn.Module,final_model_path: Path, device: torch.device,
    config: TrainConfig) -> None:
    """Sauvegarde un artefact final du modele, reutilisable pour inference.

    Args:
        model (torch.nn.Module): Modele final entraine.
        final_model_path (Path): Chemin de sauvegarde final.
        device (torch.device): Device d'entrainement.
        config (TrainConfig): Configuration d'entrainement.

    Returns:
        None: Cette fonction ecrit le state_dict final sur disque.
    """
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "saved_device": str(device),
        "model_hparams": {
            "vocab_size": int(getattr(model, "vocab_size", 0)),
            "seq_length": int(getattr(model, "seq_length", 0)),
            "embedding_dim": int(getattr(model, "embedding_dim", 0)),
            "num_blocks": int(getattr(model, "num_blocks", 0)),
            "num_heads": int(getattr(getattr(model.blocks[0], "attention", None), "num_heads", 0))
            if hasattr(model, "blocks") and len(model.blocks) > 0 else 0,
            "dropout": float(getattr(getattr(model.embedding, "dropout", None), "p", 0.0))
            if hasattr(model, "embedding") else 0.0,
        },
        "config": {
            "epochs": int(config.epochs),
            "grad_clip": float(config.grad_clip),
            "warmup_steps": int(config.warmup_steps),
            "lr_decay_factor": float(config.lr_decay_factor),
            "use_amp": bool(config.use_amp),
            "grad_accum_steps": int(config.grad_accum_steps),
        },
    }
    torch.save(payload, final_model_path)


def _apply_cpu_thread_limits(config: TrainConfig) -> None:
    """Applique un plafonnement optionnel des threads CPU PyTorch.

    Args:
        config (TrainConfig): Configuration d'entrainement.

    Returns:
        None: Cette fonction met a jour la config runtime de PyTorch.
    """
    #& Limite optionnelle des threads CPU pour reduire l'usage processeur.
    if config.max_cpu_threads is not None and config.max_cpu_threads > 0:
        torch.set_num_threads(int(config.max_cpu_threads))
    if (config.max_cpu_interop_threads is not None and config.max_cpu_interop_threads > 0):
        torch.set_num_interop_threads(int(config.max_cpu_interop_threads))

    if config.max_cpu_threads is not None or config.max_cpu_interop_threads is not None:
        print(f"[train] cpu threads: intra={torch.get_num_threads()} inter={torch.get_num_interop_threads()}")


def _resolve_steps_per_epoch(dataloader, max_steps_per_epoch: int | None) -> int:
    """Calcule le nombre de steps utilises par epoch.

    Args:
        dataloader: DataLoader d'entrainement.
        max_steps_per_epoch (int | None): Limite optionnelle par epoch.

    Returns:
        int: Nombre de steps effectifs par epoch.
    """
    if max_steps_per_epoch is None:
        return len(dataloader)
    return min(len(dataloader), max_steps_per_epoch)


def _build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig, total_steps: int):
    """Construit le scheduler LR (warmup + cosine decay).

    Args:
        optimizer (torch.optim.Optimizer): Optimizer cible.
        config (TrainConfig): Configuration d'entrainement.
        total_steps (int): Nombre total de steps prevus.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Scheduler configure.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_lr_with_warmup(
            step,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr_factor=config.lr_decay_factor,
        ),
    )


def _build_grad_scaler(device: torch.device, use_amp: bool) -> torch.amp.GradScaler:
    """Construit le scaler AMP compatible avec l'API torch.amp moderne.

    Args:
        device (torch.device): Device d'execution courant.
        use_amp (bool): Active ou non l'AMP.

    Returns:
        torch.amp.GradScaler: Scaler configure.
    """
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.GradScaler(amp_device, enabled=use_amp)


def _run_forward_loss(
    model: torch.nn.Module, loss_fn, x: torch.Tensor, y: torch.Tensor, config: TrainConfig, 
    device: torch.device, use_amp: bool) -> torch.Tensor:
    """Execute le forward et retourne une loss normalisee par accumulation.

    Args:
        model (torch.nn.Module): Modele GPT.
        loss_fn: Fonction de loss autoregressive.
        x (torch.Tensor): Batch input.
        y (torch.Tensor): Batch target.
        config (TrainConfig): Configuration d'entrainement.
        device (torch.device): Device courant.
        use_amp (bool): Active l'autocast AMP.

    Returns:
        torch.Tensor: Loss normalisee (division par grad_accum_steps).
    """
    #~ La division permet de conserver la meme echelle de gradient que
    #~ sans accumulation, meme si on empile plusieurs mini-batches.
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = model(x)
        return loss_fn(logits, y) / config.grad_accum_steps


def _optimizer_step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, scaler: torch.amp.GradScaler,
    grad_clip: float) -> float:
    """Applique clip gradients + step optimizer + step scheduler.

    Args:
        model (torch.nn.Module): Modele entraine.
        optimizer (torch.optim.Optimizer): Optimizer courant.
        scheduler: Scheduler de learning rate.
        scaler (torch.amp.GradScaler): Scaler AMP.
        grad_clip (float): Valeur max de clipping, <=0 desactive.

    Returns:
        float: Norme de gradient mesuree apres unscale.
    """
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        grad_norm = 0.0

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


def _maybe_save_periodic_checkpoint(config: TrainConfig, global_step: int, running_loss: float,
    steps: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler,
    epoch: int, ckpt_path: Path, history: dict, scaler: torch.amp.GradScaler) -> None:
    """Sauvegarde un checkpoint intermediaire selon la frequence configuree.

    Args:
        config (TrainConfig): Configuration d'entrainement.
        global_step (int): Step global courant.
        running_loss (float): Somme des losses courantes.
        steps (int): Nombre de steps cumules sur l'epoch.
        model (torch.nn.Module): Modele courant.
        optimizer (torch.optim.Optimizer): Optimizer courant.
        scheduler: Scheduler courant.
        epoch (int): Epoch courante.
        ckpt_path (Path): Chemin de checkpoint.
        history (dict): Historique d'entrainement.
        scaler (torch.amp.GradScaler): Scaler AMP a sauvegarder.

    Returns:
        None: Cette fonction ecrit sur disque si la condition est verifiee.
    """
    if config.checkpoint_every_steps <= 0:
        return
    if global_step % config.checkpoint_every_steps != 0:
        return

    avg_running = running_loss / max(steps, 1)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        step=global_step,
        avg_loss=avg_running,
        checkpoint_path=ckpt_path,
        epoch_losses=history["epoch_losses"],
        scaler=scaler,
    )
    print(f"[train] periodic checkpoint saved (step={global_step}): {ckpt_path}")


def train_model(model: torch.nn.Module, dataloader, optimizer: torch.optim.Optimizer, loss_fn, 
                config: TrainConfig) -> dict:
    """Lance la boucle complete d'entrainement supervise.

    Args:
        model (torch.nn.Module): Modele a entrainer.
        dataloader: DataLoader fournissant les batches (x, y).
        optimizer (torch.optim.Optimizer): Optimizer principal.
        loss_fn: Fonction de loss.
        config (TrainConfig): Hyperparametres et options de runtime.

    Returns:
        dict: Historique d'entrainement (losses, device, chemin checkpoint).
    """
    device = resolve_device(config.device)
    _apply_cpu_thread_limits(config)

    model.to(device)
    model.train()

    #* Preparation scheduler / AMP / checkpoints
    steps_per_epoch = _resolve_steps_per_epoch(dataloader, config.max_steps_per_epoch)
    total_steps = steps_per_epoch * config.epochs
    scheduler = _build_scheduler(optimizer, config, total_steps)

    #! AMP actif seulement sur GPU dans cette implementation.
    use_amp = config.use_amp and device.type == "cuda"
    scaler = _build_grad_scaler(device, use_amp)

    ckpt_path = Path(config.checkpoint_dir) / config.checkpoint_name
    final_model_path = Path(config.final_model_dir) / config.final_model_name
    history = {
        "epoch_losses": [],
        "checkpoint_path": str(ckpt_path),
        "final_model_path": str(final_model_path),
        "device": str(device),
    }

    start_epoch, global_step, restored_epoch_losses = load_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=ckpt_path,
        resume=config.resume_from_checkpoint,
        scaler=scaler,
    )
    history["epoch_losses"].extend(restored_epoch_losses)

    #* Boucle d'entrainement principale
    for epoch in range(start_epoch, config.epochs):
        running_loss = 0.0
        steps = 0
        grad_norm = 0.0
        start_t = time.time()

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(dataloader):
            if config.max_steps_per_epoch is not None and batch_idx >= config.max_steps_per_epoch:
                break

            x = x.to(device)
            y = y.to(device)

            #~ Equivalent batch size = batch_size * grad_accum_steps.
            is_last_accum = (batch_idx + 1) % config.grad_accum_steps == 0

            loss = _run_forward_loss(
                model=model,
                loss_fn=loss_fn,
                x=x,
                y=y,
                config=config,
                device=device,
                use_amp=use_amp,
            )

            scaler.scale(loss).backward()

            if is_last_accum:
                grad_norm = _optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    grad_clip=config.grad_clip,
                )
                global_step += 1
                _maybe_save_periodic_checkpoint(
                    config=config,
                    global_step=global_step,
                    running_loss=running_loss,
                    steps=steps,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    ckpt_path=ckpt_path,
                    history=history,
                    scaler=scaler,
                )

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
            epoch_losses=history["epoch_losses"],
            scaler=scaler,
        )
        print(f"[train] checkpoint saved: {ckpt_path}")

    if config.save_final_model_state:
        save_final_model_state(
            model=model,
            final_model_path=final_model_path,
            device=device,
            config=config,
        )
        print(f"[train] final model state saved: {final_model_path}")

    return history