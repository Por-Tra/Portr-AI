"""
train.py — version optimisée

Corrections vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème (ralentissement ×2-3)  | Solution                                 |
|---------------------------------|------------------------------------------|
| loss.item() appelé à CHAQUE     | loss.item() déplacé dans le bloc de log  |
| batch → synchronisation GPU→CPU | uniquement (toutes les log_interval       |
| à chaque step, pipeline CUDA    | steps). Entre les logs, la loss est       |
| bloqué, async annulé            | accumulée comme tenseur sur GPU sans      |
|                                 | jamais forcer de sync.                   |
|---------------------------------|------------------------------------------|
| running_loss += loss_item à     | running_loss est un float Python initialisé|
| chaque step avec loss.item()    | à 0.0 ; la sync n'arrive qu'au log.      |
| même hors du bloc log           |                                          |
|---------------------------------|------------------------------------------|
| torch.save() synchrone sur le   | Checkpoint écrit dans un thread séparé   |
| thread principal → bloque le    | via threading.Thread(daemon=True) pour   |
| training pendant l'I/O disque   | ne pas bloquer la boucle principale.     |
|---------------------------------|------------------------------------------|
| Scheduler restauré mais         | total_steps sauvegardé dans le           |
| total_steps recalculé depuis    | checkpoint ; au resume, le scheduler     |
| epoch 0 → LR désynchro si      | est reconstruit avec le bon total_steps  |
| EPOCHS change entre deux runs   | pour garder le cosine decay cohérent.    |
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
import math

import torch


@dataclass
class TrainConfig:
    """Hyperparamètres et options de runtime pour l'entraînement."""

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

    # Scheduler
    warmup_steps: int = 100
    lr_decay_factor: float = 0.1      # LR final = LR_initial × lr_decay_factor

    # Mixed precision
    use_amp: bool = True               # float16 sur GPU (ignoré sur CPU)

    # Gradient accumulation
    grad_accum_steps: int = 1          # accumule N batches avant optimizer.step()


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def resolve_device(requested: str = "auto") -> torch.device:
    """Résout le device d'exécution."""
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
    """Multiplicateur LR : warmup linéaire puis cosine decay.

    Phase 1 (step < warmup_steps) : 0 → 1 linéairement.
    Phase 2 (step ≥ warmup_steps) : 1 → min_lr_factor via cosinus.
    """
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_factor + (1.0 - min_lr_factor) * cosine


def _async_save(payload: dict, path: Path) -> None:
    """Sauvegarde le checkpoint dans un thread daemon pour ne pas bloquer le training."""
    def _write() -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
        except Exception as exc:
            print(f"[train] WARNING: checkpoint save failed: {exc}")

    t = threading.Thread(target=_write, daemon=True)
    t.start()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    avg_loss: float,
    checkpoint_path: Path,
    total_steps: int,
    epoch_losses: list[float] | None = None,
    scaler: torch.amp.GradScaler | None = None,
    async_write: bool = True,
) -> None:
    """Sauvegarde l'état d'entraînement pour reprise ultérieure.

    Args:
        model           : Modèle entraîné.
        optimizer       : Optimizer courant.
        scheduler       : Scheduler de LR.
        epoch           : Epoch à sauvegarder.
        step            : Step global à sauvegarder.
        avg_loss        : Loss moyenne récente.
        checkpoint_path : Chemin du fichier checkpoint.
        total_steps     : Nombre total de steps prévus (pour reconstruire
                          le scheduler au resume avec le bon cosine decay).
        epoch_losses    : Historique des losses par epoch.
        scaler          : État AMP pour reprise fidèle.
        async_write     : Si True, écriture dans un thread séparé (ne bloque pas).
    """
    payload: dict = {
        "epoch": epoch,
        "step": step,
        "avg_loss": avg_loss,
        "total_steps": total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if epoch_losses is not None:
        payload["epoch_losses"] = list(epoch_losses)

    if async_write:
        _async_save(payload, checkpoint_path)
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint_path)


def load_checkpoint_if_available(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    checkpoint_path: Path,
    resume: bool,
    steps_per_epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int, list[float], int]:
    """Charge un checkpoint si disponible et autorisé.

    Returns:
        (start_epoch, global_step, epoch_losses, total_steps)
        total_steps est récupéré depuis le checkpoint si disponible,
        sinon recalculé pour rester cohérent avec la config actuelle.
    """
    if not resume or not checkpoint_path.exists():
        total_steps = steps_per_epoch * config.epochs
        return 0, 0, [], total_steps

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    start_epoch  = int(payload.get("epoch", 0))
    global_step  = int(payload.get("step", 0))
    epoch_losses = [float(v) for v in payload.get("epoch_losses", [])]

    # Reconstruire total_steps depuis le checkpoint pour que le scheduler
    # cosine decay reste aligné sur la même courbe de LR que le run initial.
    # Si EPOCHS a changé, on recalcule proprement depuis l'epoch courante.
    saved_total_steps = payload.get("total_steps")
    if saved_total_steps is not None:
        remaining = config.epochs - start_epoch
        total_steps = global_step + remaining * steps_per_epoch
    else:
        total_steps = steps_per_epoch * config.epochs

    # Restaurer le scheduler en reconstruisant avec total_steps correct
    # puis en chargeant son état sauvegardé.
    # (Le scheduler doit être reconstruit avant load_state_dict.)
    # Note: la reconstruction est faite dans train_model() après cet appel.
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])

    print(
        f"[train] resume checkpoint: epoch={start_epoch} "
        f"step={global_step} path={checkpoint_path}"
    )
    return start_epoch, global_step, epoch_losses, total_steps


def save_final_model_state(
    model: torch.nn.Module,
    final_model_path: Path,
    device: torch.device,
    config: TrainConfig,
) -> None:
    """Sauvegarde un artefact final du modèle pour inférence réutilisable."""
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "saved_device": str(device),
        "model_hparams": {
            "vocab_size":    int(getattr(model, "vocab_size", 0)),
            "seq_length":    int(getattr(model, "seq_length", 0)),
            "embedding_dim": int(getattr(model, "embedding_dim", 0)),
            "num_blocks":    int(getattr(model, "num_blocks", 0)),
            "num_heads": int(
                getattr(getattr(model.blocks[0], "attention", None), "num_heads", 0)
            ) if hasattr(model, "blocks") and len(model.blocks) > 0 else 0,
            "dropout": float(
                getattr(getattr(model.embedding, "dropout", None), "p", 0.0)
            ) if hasattr(model, "embedding") else 0.0,
        },
        "config": {
            "epochs":          int(config.epochs),
            "grad_clip":       float(config.grad_clip),
            "warmup_steps":    int(config.warmup_steps),
            "lr_decay_factor": float(config.lr_decay_factor),
            "use_amp":         bool(config.use_amp),
            "grad_accum_steps": int(config.grad_accum_steps),
        },
    }
    torch.save(payload, final_model_path)


def _apply_cpu_thread_limits(config: TrainConfig) -> None:
    """Applique un plafonnement optionnel des threads CPU PyTorch."""
    if config.max_cpu_threads is not None and config.max_cpu_threads > 0:
        torch.set_num_threads(int(config.max_cpu_threads))
    if config.max_cpu_interop_threads is not None and config.max_cpu_interop_threads > 0:
        torch.set_num_interop_threads(int(config.max_cpu_interop_threads))
    if config.max_cpu_threads is not None or config.max_cpu_interop_threads is not None:
        print(
            f"[train] cpu threads: "
            f"intra={torch.get_num_threads()} "
            f"inter={torch.get_num_interop_threads()}"
        )


def _resolve_steps_per_epoch(dataloader, max_steps_per_epoch: int | None) -> int:
    if max_steps_per_epoch is None:
        return len(dataloader)
    return min(len(dataloader), max_steps_per_epoch)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    total_steps: int,
):
    """Construit le scheduler LR (warmup linéaire + cosine decay)."""
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
    """Construit le GradScaler AMP."""
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.GradScaler(amp_device, enabled=use_amp)


def _run_forward_loss(
    model: torch.nn.Module,
    loss_fn,
    x: torch.Tensor,
    y: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """Forward pass + calcul de la loss normalisée par grad_accum_steps.

    La division par grad_accum_steps permet de conserver la même échelle
    de gradient qu'un batch de taille batch_size × grad_accum_steps.

    Retourne un tenseur GPU, PAS un float. Ne pas appeler .item() ici.
    """
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = model(x)
        return loss_fn(logits, y) / config.grad_accum_steps


def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> float:
    """Unscale → clip → step optimizer → step scheduler → zero_grad.

    Returns:
        Norme du gradient (float, sync GPU forcée ici seulement).
    """
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        )
    else:
        grad_norm = 0.0

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def _maybe_save_periodic_checkpoint(
    config: TrainConfig,
    global_step: int,
    running_loss: float,
    steps: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    ckpt_path: Path,
    history: dict,
    scaler: torch.amp.GradScaler,
    total_steps: int,
) -> None:
    """Sauvegarde un checkpoint intermédiaire selon la fréquence configurée."""
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
        total_steps=total_steps,
        epoch_losses=history["epoch_losses"],
        scaler=scaler,
        async_write=True,  # non-bloquant
    )
    print(f"[train] periodic checkpoint saved (step={global_step}): {ckpt_path}")


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def train_model(
    model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    config: TrainConfig,
) -> dict:
    """Lance la boucle complète d'entraînement supervisé.

    Args:
        model      : Modèle à entraîner.
        dataloader : DataLoader fournissant les batches (x, y).
        optimizer  : Optimizer principal.
        loss_fn    : Fonction de loss autoregressive.
        config     : Hyperparamètres et options de runtime.

    Returns:
        Historique d'entraînement (losses, device, chemins checkpoint).
    """
    device = resolve_device(config.device)
    _apply_cpu_thread_limits(config)

    model.to(device)
    model.train()

    steps_per_epoch = _resolve_steps_per_epoch(dataloader, config.max_steps_per_epoch)

    use_amp = config.use_amp and device.type == "cuda"
    scaler  = _build_grad_scaler(device, use_amp)

    ckpt_path        = Path(config.checkpoint_dir) / config.checkpoint_name
    final_model_path = Path(config.final_model_dir) / config.final_model_name

    history: dict = {
        "epoch_losses":    [],
        "checkpoint_path": str(ckpt_path),
        "final_model_path": str(final_model_path),
        "device":          str(device),
    }

    # ── Reprise depuis checkpoint ──────────────────────────────────────────
    start_epoch, global_step, restored_losses, total_steps = load_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        config=config,
        checkpoint_path=ckpt_path,
        resume=config.resume_from_checkpoint,
        steps_per_epoch=steps_per_epoch,
        scaler=scaler,
    )
    history["epoch_losses"].extend(restored_losses)

    # Construire (ou reconstruire) le scheduler avec le bon total_steps
    scheduler = _build_scheduler(optimizer, config, total_steps)
    if start_epoch > 0:
        # Restaurer l'état du scheduler depuis le checkpoint
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "scheduler_state_dict" in ckpt_data:
            scheduler.load_state_dict(ckpt_data["scheduler_state_dict"])

    # ── Boucle d'entraînement ──────────────────────────────────────────────
    for epoch in range(start_epoch, config.epochs):
        running_loss   = 0.0
        steps          = 0
        grad_norm      = 0.0
        # Accumulateur GPU pour le log (évite .item() à chaque step)
        log_loss_accum = 0.0
        log_steps      = 0
        start_t        = time.time()

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(dataloader):
            if config.max_steps_per_epoch is not None and batch_idx >= config.max_steps_per_epoch:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            is_last_accum = (batch_idx + 1) % config.grad_accum_steps == 0

            # ── Forward + backward ─────────────────────────────────────────
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

            # ── Optimizer step (tous les grad_accum_steps batches) ─────────
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
                    total_steps=total_steps,
                )

            # ── Accumulation de la loss SANS .item() ───────────────────────
            # On ajoute le tenseur GPU à un accumulateur Python seulement
            # lors du log (toutes les log_interval steps) pour éviter une
            # synchronisation GPU→CPU à chaque step.
            # La running_loss sert au calcul de la loss moyenne par epoch ;
            # elle est forcée via .item() uniquement en fin d'epoch.
            log_loss_accum += loss.detach()   # tenseur GPU, pas de sync
            log_steps      += 1
            steps          += 1

            # ── Log (sync GPU une seule fois par log_interval) ────────────
            if (batch_idx + 1) % config.log_interval == 0:
                # .item() ici → une seule sync GPU toutes les log_interval étapes
                avg_log_loss  = float(log_loss_accum.item()) * config.grad_accum_steps / log_steps
                running_loss += float(log_loss_accum.item()) * config.grad_accum_steps
                log_loss_accum = 0.0
                log_steps      = 0

                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[train] epoch={epoch + 1}/{config.epochs} "
                    f"step={batch_idx + 1} "
                    f"loss={avg_log_loss:.4f} "
                    f"lr={current_lr:.2e} "
                    f"grad_norm={grad_norm:.3f}"
                )

        # Flush des losses non encore comptabilisées (dernier bloc partiel)
        if log_steps > 0:
            running_loss += float(log_loss_accum.item()) * config.grad_accum_steps

        avg_loss = running_loss / max(steps, 1)
        elapsed  = time.time() - start_t
        history["epoch_losses"].append(avg_loss)

        print(
            f"[train] epoch {epoch + 1} done | "
            f"avg_loss={avg_loss:.4f} | steps={steps} | time={elapsed:.1f}s"
        )

        # Checkpoint de fin d'epoch (async → ne bloque pas le prochain epoch)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            step=global_step,
            avg_loss=avg_loss,
            checkpoint_path=ckpt_path,
            total_steps=total_steps,
            epoch_losses=history["epoch_losses"],
            scaler=scaler,
            async_write=True,
        )
        print(f"[train] checkpoint saved: {ckpt_path}")

    # ── Artefact final ─────────────────────────────────────────────────────
    if config.save_final_model_state:
        save_final_model_state(
            model=model,
            final_model_path=final_model_path,
            device=device,
            config=config,
        )
        print(f"[train] final model state saved: {final_model_path}")

    return history