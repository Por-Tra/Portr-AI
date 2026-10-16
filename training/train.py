# -*- coding: utf-8 -*-
"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : train.py

description du script:
    Script d'entraînement du modèle GPT sur le dataset préparé.

"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import threading
import time

import torch


@dataclass
class TrainConfig:
    """Hyperparamètres et options de runtime pour l'entraînement."""

    epochs: int = 1
    device: str = "auto"
    grad_clip: float = 1.0
    log_interval: int = 20
    max_steps_per_epoch: int | None = None   #& en micro-batches
    checkpoint_dir: str = "artifacts/checkpoints"
    checkpoint_name: str = "gpt_latest.pt"
    resume_from_checkpoint: bool = True
    checkpoint_every_steps: int = 200        #& en optimizer steps
    save_final_model_state: bool = True
    final_model_dir: str = "artifacts/models"
    final_model_name: str = "gpt_final_state.pt"
    max_cpu_threads: int | None = None
    max_cpu_interop_threads: int | None = None
    seed: int = 1337                         #& ordre de shuffle reproductible (reprise exacte)

    #* Scheduler
    warmup_steps: int = 100                  #& en optimizer steps
    warmup_ratio: float | None = None        #& si défini, remplace warmup_steps (ex: 0.02 = 2%)
    lr_decay_factor: float = 0.1             #& LR final = LR_initial x lr_decay_factor

    #* Mixed precision
    use_amp: bool = True                     #& ignoré sur CPU
    amp_dtype: str = "auto"                  #& "auto" (bf16 si supporté sinon fp16), "bf16", "fp16"

    #* Gradient accumulation
    grad_accum_steps: int = 1                #& accumule N batches avant optimizer.step()

    #* Validation
    eval_interval: int = 0                   #& en optimizer steps (0 = seulement en fin d'epoch)
    eval_max_batches: int | None = 200

    #* torch.compile (nécessite Triton : Linux, ou triton-windows sous Windows)
    compile_model: bool = False


#* ---------------------------------------------------------------------------
#*                          Helpers internes
#* ---------------------------------------------------------------------------

def resolve_device(requested: str = "auto"):
    """ Résout le device cible pour l'entraînement.

    Args:
        requested (str, optional): Le device demandé. Defaults to "auto".

    Returns:
        torch.device: Le device résolu.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_amp_dtype(device: torch.device, requested: str):
    """ Résout le type de données pour l'AMP (Automatic Mixed Precision) en fonction du device et de la demande.
    bf16 est préféré sur les GPU récents, sinon fp16. Sur CPU, AMP n'est pas supporté.

    Args:
        device (torch.device): Le device cible pour l'entraînement.
        requested (str): Le type de données demandé.

    Returns:
        torch.dtype: Le type de données résolu.
    """
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _cosine_lr_with_warmup(step: int, warmup_steps: int, total_steps: int, min_lr_factor: float):
    """ Calcul du facteur de learning rate pour un scheduler avec warmup linéaire puis décroissance cosinus.

    Args:
        step (int): Numéro de l'optimizer step courant.
        warmup_steps (int): Nombre d'epochs de warmup.
        total_steps (int): Nombre total d'epochs.
        min_lr_factor (float): Facteur de learning rate minimum.

    Returns:
        float: Le facteur de learning rate calculé.
    """
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)

    #& progress borné à 1 : sans ça le cosinus remonte au-delà de total_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_factor + (1.0 - min_lr_factor) * cosine


#* ---------------------------------------------------------------------------
#*                              Checkpoints
#* ---------------------------------------------------------------------------

_pending_save: threading.Thread | None = None


def _to_cpu_snapshot(obj):
    """ Crée une copie CPU d'un objet (tenseur, dict, list, tuple) pour sauvegarde.

    Args:
        obj (_type_): L'objet à copier sur CPU.

    Returns:
        Any: L'objet copié sur CPU.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu_snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu_snapshot(v) for v in obj)
    return obj


def _atomic_save(payload: dict, path: Path):
    """ Sauvegarde atomique d'un checkpoint : écrit dans un fichier temporaire puis renomme.

    Args:
        payload (dict): Dictionnaire contenant les informations à sauvegarder.
        path (Path): Chemin du fichier de checkpoint final.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def wait_for_pending_checkpoint():
    """Attend la fin de l'écriture asynchrone en cours (si existante)."""
    global _pending_save
    if _pending_save is not None:
        _pending_save.join()
        _pending_save = None


def _async_save(payload: dict, path: Path):
    """ Sauvegarde asynchrone d'un checkpoint dans un thread séparé.

    Args:
        payload (dict): Dictionnaire contenant les informations à sauvegarder.
        path (Path): Chemin du fichier de checkpoint final.
    """
    global _pending_save
    wait_for_pending_checkpoint()

    def _write():
        try:
            _atomic_save(payload, path)
        except Exception as exc:
            print(f"[train] WARNING: checkpoint save failed: {exc}")

    _pending_save = threading.Thread(target=_write, daemon=False)
    _pending_save.start()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, batch_in_epoch: int,step: int,
    avg_loss: float, checkpoint_path: Path, total_steps: int,
      epoch_losses: list[float] | None = None, val_losses: list[float] | None = None,
        scaler: torch.amp.GradScaler | None = None, async_write: bool = True):
    """ Sauvegarde un checkpoint d'entraînement (état du modèle, optimizer, scheduler, etc.) dans un fichier.

    Args:
        model (torch.nn.Module): Modèle à sauvegarder.
        optimizer (torch.optim.Optimizer): Optimizer à sauvegarder.
        scheduler (_type_): Scheduler à sauvegarder.
        epoch (int): Numéro de l'epoch en cours (ou suivante si l'epoch est terminée).
        batch_in_epoch (int): Nombre de micro-batches déjà consommés dans `epoch`.
        step (int): Step global (optimizer steps).
        avg_loss (float): Loss moyenne récente.
        checkpoint_path (Path): Chemin du fichier checkpoint.
        total_steps (int): Nombre total d'optimizer steps prévus.
        epoch_losses (list[float] | None, optional): Historique des losses train par epoch. Defaults to None.
        val_losses (list[float] | None, optional): Historique des losses de validation. Defaults to None.
        scaler (torch.amp.GradScaler | None, optional): État AMP pour reprise fidèle. Defaults to None.
        async_write (bool, optional): Si True, écriture dans un thread séparé. Defaults to True.
    """
    payload: dict = {
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "step": step,
        "avg_loss": avg_loss,
        "total_steps": total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch_losses": list(epoch_losses or []),
        "val_losses": list(val_losses or [])
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()

    payload = _to_cpu_snapshot(payload)

    if async_write:
        _async_save(payload, checkpoint_path)
    else:
        wait_for_pending_checkpoint()
        _atomic_save(payload, checkpoint_path)


def load_checkpoint_if_available(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, 
                                 checkpoint_path: Path, resume: bool, 
                                 scaler: torch.amp.GradScaler | None = None):
    """ Charge un checkpoint d'entraînement si disponible et compatible avec le modèle actuel.

    Args:
        model (torch.nn.Module): Modèle à restaurer.
        optimizer (torch.optim.Optimizer): Optimizer à restaurer.
        scheduler (_type_): Scheduler à restaurer.
        checkpoint_path (Path): Chemin du fichier de checkpoint.
        resume (bool): Indique si on doit reprendre à partir du checkpoint.
        scaler (torch.amp.GradScaler | None, optional): État AMP pour reprise fidèle. Defaults to None.

    Raises:
        RuntimeError: Si le checkpoint est incompatible avec le modèle actuel (ex: vocab ou architecture modifiés).

    Returns:
        tuple[int, int, int, list[float], list[float]]: Les valeurs de départ pour l'entraînement.
    """
    if not resume or not checkpoint_path.exists():
        return 0, 0, 0, [], []

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(payload["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint incompatible avec le modèle actuel ({checkpoint_path}).Le vocab ou l'architecture a probablement changé (ex: tokenizer ré-entraîné). Supprime le checkpoint ou passe RESUME_FROM_CHECKPOINT=False.") from exc
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    #& Toujours restaurer le scheduler (avant : seulement si epoch > 0,
    #& donc un checkpoint pris pendant la 1ère epoch relançait le warmup)
    if "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])

    start_epoch = int(payload.get("epoch", 0))
    batch_in_epoch = int(payload.get("batch_in_epoch", 0))
    global_step = int(payload.get("step", 0))
    epoch_losses = [float(v) for v in payload.get("epoch_losses", [])]
    val_losses = [float(v) for v in payload.get("val_losses", [])]

    print(f"[train] resume checkpoint: epoch={start_epoch} batch={batch_in_epoch} step={global_step} path={checkpoint_path}")
    return start_epoch, batch_in_epoch, global_step, epoch_losses, val_losses


def save_final_model_state(model: torch.nn.Module, final_model_path: Path, device: torch.device, config: TrainConfig):
    """ Sauvegarde l'état final du modèle après l'entraînement, avec les hyperparamètres et la configuration d'entraînement.

    Args:
        model (torch.nn.Module): Modèle à sauvegarder.
        final_model_path (Path): Chemin vers le fichier de sauvegarde du modèle final.
        device (torch.device): Device sur lequel le modèle est actuellement.
        config (TrainConfig): Configuration d'entraînement.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "saved_device": str(device),
        "model_hparams": {
            "vocab_size":    int(getattr(model, "vocab_size", 0)),
            "seq_length":    int(getattr(model, "seq_length", 0)),
            "embedding_dim": int(getattr(model, "embedding_dim", 0)),
            "num_blocks":    int(getattr(model, "num_blocks", 0)),
            "num_heads": int( getattr(getattr(model.blocks[0], "attention", None), "num_heads", 0)
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
    _atomic_save(_to_cpu_snapshot(payload), final_model_path)


#* ---------------------------------------------------------------------------
#*                          Étapes de la boucle
#* ---------------------------------------------------------------------------

def _apply_cpu_thread_limits(config: TrainConfig):
    """ Applique les limites de threads CPU pour l'entraînement, si spécifiées dans la configuration.

    Args:
        config (TrainConfig): Configuration d'entraînement contenant les limites de threads CPU.
    """
    if config.max_cpu_threads is not None and config.max_cpu_threads > 0:
        torch.set_num_threads(int(config.max_cpu_threads))
    if config.max_cpu_interop_threads is not None and config.max_cpu_interop_threads > 0:
        try:
            torch.set_num_interop_threads(int(config.max_cpu_interop_threads))
        except RuntimeError:
            pass  #& ne peut être appelé qu'une fois par process
    if config.max_cpu_threads is not None or config.max_cpu_interop_threads is not None:
        print(
            f"[train] cpu threads: "
            f"intra={torch.get_num_threads()} "
            f"inter={torch.get_num_interop_threads()}"
        )


def _resolve_steps_per_epoch(dataloader, max_steps_per_epoch: int | None):
    """ Résout le nombre d'optimizer steps par epoch, limité par max_steps_per_epoch si défini.

    Args:
        dataloader (_type_): Dataloader fournissant les batches d'entraînement.
        max_steps_per_epoch (int | None): Nombre maximum d'optimizer steps par epoch. Si None, utilise la taille du dataloader.

    Returns:
        int: Nombre d'optimizer steps par epoch (limité par max_steps_per_epoch si défini).
    """
    if max_steps_per_epoch is None:
        return len(dataloader)
    return min(len(dataloader), max_steps_per_epoch)


def _build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, lr_decay_factor: float):
    """ construit un scheduler de learning rate avec warmup linéaire puis décroissance cosinus.

    Args:
        optimizer (torch.optim.Optimizer): L'optimizer pour lequel construire le scheduler.
        warmup_steps (int): Nombre de steps de warmup.
        total_steps (int): Nombre total de steps.
        lr_decay_factor (float): Facteur de décroissance du learning rate.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: Le scheduler de learning rate construit.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_lr_with_warmup(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_factor=lr_decay_factor
        )
    )


def _run_forward_loss(model: torch.nn.Module, loss_fn, x: torch.Tensor, y: torch.Tensor,
                       accum_divisor: int,device: torch.device, use_amp: bool, amp_dtype: torch.dtype):
    """ Exécute le forward pass du modèle et calcule la loss, avec support pour l'AMP (Automatic Mixed Precision).

    Args:
        model (torch.nn.Module): Modèle à entraîner.
        loss_fn (_type_): Fonction de loss à utiliser.
        x (torch.Tensor): Données d'entrée.
        y (torch.Tensor): Données de sortie attendues.
        accum_divisor (int): Diviseur pour l'accumulation des gradients.
        device (torch.device): Device sur lequel exécuter le calcul.
        use_amp (bool): Indique si l'AMP est utilisé.
        amp_dtype (torch.dtype): Type de données à utiliser pour l'AMP.
        amp_dtype (torch.dtype): Type de données à utiliser pour l'AMP.

    Returns:
        torch.Tensor: La loss calculée pour le batch courant, divisée par accum_divisor.
    """
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        logits = model(x)
        return loss_fn(logits, y) / accum_divisor


def _optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler,
                     scaler: torch.amp.GradScaler, grad_clip: float):
    """ Effectue un step de l'optimizer avec support pour l'AMP (Automatic Mixed Precision)
      et le clipping des gradients.

    Args:
        model (torch.nn.Module): Modèle à entraîner.
        optimizer (torch.optim.Optimizer): optimizer à utiliser pour la mise à jour des poids.
        scheduler (_type_): scheduler de learning rate à mettre à jour après le step.
        scaler (torch.amp.GradScaler): Scaler pour l'AMP, utilisé pour la mise à l'échelle des gradients.
        grad_clip (float): Valeur maximale pour le clipping des gradients. Si <= 0, aucun clipping n'est appliqué.

    Returns:
        torch.Tensor | None: La norme des gradients après le clipping,
          ou None si aucun clipping n'a été appliqué.
    """
    grad_norm = None
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader, loss_fn, device: torch.device, use_amp: bool,
              amp_dtype: torch.dtype, max_batches: int | None = None):
    """ Évalue le modèle sur un ensemble de validation et retourne la loss moyenne.

    Args:
        model (torch.nn.Module): Modèle à évaluer.
        dataloader (_type_): Dataloader fournissant les batches de validation.
        loss_fn (_type_): Fonction de loss à utiliser pour l'évaluation.
        device (torch.device): Device sur lequel exécuter l'évaluation.
        use_amp (bool): Indique si l'AMP est utilisé.
        amp_dtype (torch.dtype): Type de données à utiliser pour l'AMP.
        max_batches (int | None, optional): Nombre maximum de batches à évaluer. Defaults to None.

    Returns:
        float: La loss moyenne sur l'ensemble de validation.
    """
    was_training = model.training
    model.eval()
    total = torch.zeros((), device=device)
    count = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            total += loss_fn(model(x), y).float()
        count += 1
    if was_training:
        model.train()
    return float(total.item()) / max(count, 1)


def _seed_epoch(dataloader, seed: int, epoch: int, start_batch: int):
    """ Configure le sampler du dataloader pour l'epoch courante, en utilisant un seed 
    reproductible et en sautant les batches déjà consommés.

    Args:
        dataloader (_type_): Dataloader à configurer.
        seed (int): Seed de base pour la génération aléatoire.
        epoch (int): Numéro de l'epoch courante.
        start_batch (int): Numéro du batch à partir duquel commencer l'epoch (pour reprise).

    Returns:
        bool: True si le sampler a été configuré avec succès, False sinon.
    """
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        batch_size = dataloader.batch_size or 1
        try:
            sampler.set_epoch(epoch, start_index=start_batch * batch_size)
            return True
        except TypeError:
            sampler.set_epoch(epoch)
    return False


#* ---------------------------------------------------------------------------
#* Boucle principale
#* ---------------------------------------------------------------------------

def train_model(model: torch.nn.Module, dataloader, optimizer: torch.optim.Optimizer,
                 loss_fn, config: TrainConfig, val_dataloader=None, eval_loss_fn=None):
    """ Entraîne le modèle sur les données fournies, avec support pour l'AMP, le checkpointing, et la validation.

    Args:
        model (torch.nn.Module): modèle à entraîner.
        dataloader (_type_): Dataloader fournissant les batches d'entraînement.
        optimizer (torch.optim.Optimizer): optimizer à utiliser pour la mise à jour des poids.
        loss_fn (_type_): Fonction de loss à utiliser pour l'entraînement.
        config (TrainConfig): Configuration d'entraînement contenant les hyperparamètres et options de runtime.
        val_dataloader (_type_, optional): Dataloader fournissant les batches de validation. Defaults to None.
        eval_loss_fn (_type_, optional): Fonction de loss à utiliser pour l'évaluation. Defaults to None.

    Returns:
        dict: Historique de l'entraînement contenant les losses par epoch, les losses de validation, et les chemins des checkpoints et du modèle final.
    """
    device = resolve_device(config.device)
    _apply_cpu_thread_limits(config)

    model.to(device)
    model.train()

    #& Le module compilé ne sert qu'au forward ; on sauvegarde toujours `model`
    #& (le state_dict d'un module compilé a des clés préfixées "_orig_mod.").
    forward_model = torch.compile(model) if config.compile_model else model
    eval_loss_fn = eval_loss_fn or loss_fn

    accum = max(1, int(config.grad_accum_steps))
    steps_per_epoch = _resolve_steps_per_epoch(dataloader, config.max_steps_per_epoch)
    #& Le scheduler avance une fois par optimizer step, pas par micro-batch
    optim_steps_per_epoch = math.ceil(steps_per_epoch / accum)
    total_steps = optim_steps_per_epoch * config.epochs

    warmup_steps = (
        int(total_steps * config.warmup_ratio)
        if config.warmup_ratio is not None else int(config.warmup_steps)
    )
    config.warmup_steps = warmup_steps

    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(device, config.amp_dtype)
    #& GradScaler uniquement utile en fp16 (bf16 a la même plage que fp32)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

    print(
        f"[train] device={device} | amp={'off' if not use_amp else str(amp_dtype).replace('torch.', '')} "
        f"| micro-batches/epoch={steps_per_epoch} | optimizer steps/epoch={optim_steps_per_epoch} "
        f"| total_steps={total_steps} | warmup={warmup_steps}"
    )

    ckpt_path        = Path(config.checkpoint_dir) / config.checkpoint_name
    final_model_path = Path(config.final_model_dir) / config.final_model_name

    history: dict = {
        "epoch_losses": [],
        "val_losses": [],
        "checkpoint_path": str(ckpt_path),
        "final_model_path": str(final_model_path),
        "device": str(device)
    }

    scheduler = _build_scheduler(optimizer, warmup_steps, total_steps, config.lr_decay_factor)

    # * Reprise depuis checkpoint *
    start_epoch, resume_batch, global_step, restored_losses, restored_val = load_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=ckpt_path,
        resume=config.resume_from_checkpoint,
        scaler=scaler
    )
    history["epoch_losses"].extend(restored_losses)
    history["val_losses"].extend(restored_val)

    def _run_eval(tag: str): #& fonction interne pour évaluation périodique
        if val_dataloader is None:
            return
        val_loss = evaluate(model, val_dataloader, eval_loss_fn, device, use_amp, amp_dtype, config.eval_max_batches)
        history["val_losses"].append(val_loss)
        ppl = math.exp(min(val_loss, 50.0))
        print(f"[eval] {tag} step={global_step} val_loss={val_loss:.4f} val_ppl={ppl:.2f}")

    # * Boucle d'entraînement *
    for epoch in range(start_epoch, config.epochs):
        skip = resume_batch if epoch == start_epoch else 0
        if skip >= steps_per_epoch:
            continue
        sampler_skips = _seed_epoch(dataloader, config.seed, epoch, skip)
        if skip:
            print(f"[train] reprise epoch {epoch + 1} au micro-batch {skip}")

        running_loss = 0.0
        steps = 0
        grad_norm = None
        #& Accumulateur GPU pour le log (évite .item() à chaque step)
        log_loss_accum = torch.zeros((), device=device)
        log_steps = 0
        start_t = time.time()

        optimizer.zero_grad(set_to_none=True)

        data_iter = iter(dataloader)
        if skip and not sampler_skips:
            #& Fallback : sampler non reproductible -> on consomme les batches
            for _ in range(skip):
                next(data_iter)

        for batch_idx, (x, y) in enumerate(data_iter, start=skip):
            if batch_idx >= steps_per_epoch:
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            #& Le dernier groupe d'accumulation de l'epoch peut être incomplet :
            #& on normalise par sa taille réelle et on fait quand même le step.
            group_start = (batch_idx // accum) * accum
            group_size = min(accum, steps_per_epoch - group_start)
            is_last_accum = (batch_idx + 1 - group_start) == group_size

            # * Forward + backward *
            loss = _run_forward_loss(
                model=forward_model,
                loss_fn=loss_fn,
                x=x,
                y=y,
                accum_divisor=group_size,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )
            scaler.scale(loss).backward()

            #& Pas de .item() ici : accumulation sur GPU, sync uniquement au log
            log_loss_accum += loss.detach() * group_size
            log_steps += 1
            steps += 1

            # * Optimizer step (tous les grad_accum_steps batches) *
            if is_last_accum:
                grad_norm = _optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    grad_clip=config.grad_clip
                )
                global_step += 1

                if config.checkpoint_every_steps > 0 and global_step % config.checkpoint_every_steps == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        batch_in_epoch=batch_idx + 1,
                        step=global_step,
                        avg_loss=running_loss / max(steps - log_steps, 1),
                        checkpoint_path=ckpt_path,
                        total_steps=total_steps,
                        epoch_losses=history["epoch_losses"],
                        val_losses=history["val_losses"],
                        scaler=scaler,
                        async_write=True
                    )
                    print(f"[train] periodic checkpoint saved (step={global_step}): {ckpt_path}")

                if config.eval_interval > 0 and global_step % config.eval_interval == 0:
                    _run_eval("periodic")

            # * Log (sync GPU une seule fois par log_interval) *
            if (batch_idx + 1) % config.log_interval == 0:
                log_sum = float(log_loss_accum.item())
                running_loss += log_sum
                avg_log_loss = log_sum / max(log_steps, 1)
                log_loss_accum.zero_()
                log_steps = 0

                current_lr = scheduler.get_last_lr()[0]
                gn = float(grad_norm.item()) if grad_norm is not None else 0.0
                print(
                    f"[train] epoch={epoch + 1}/{config.epochs} "
                    f"batch={batch_idx + 1}/{steps_per_epoch} "
                    f"step={global_step}/{total_steps} "
                    f"loss={avg_log_loss:.4f} "
                    f"lr={current_lr:.2e} "
                    f"grad_norm={gn:.3f}"
                )

        #& Flush des losses non encore comptabilisées (dernier bloc partiel)
        if log_steps > 0:
            running_loss += float(log_loss_accum.item())

        avg_loss = running_loss / max(steps, 1)
        elapsed = time.time() - start_t
        history["epoch_losses"].append(avg_loss)

        print(f"[train] epoch {epoch + 1} done | avg_loss={avg_loss:.4f} | batches={steps} | time={elapsed:.1f}s")
        _run_eval(f"epoch {epoch + 1}")

        #& Checkpoint de fin d'epoch : la reprise commence à l'epoch suivante
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            batch_in_epoch=0,
            step=global_step,
            avg_loss=avg_loss,
            checkpoint_path=ckpt_path,
            total_steps=total_steps,
            epoch_losses=history["epoch_losses"],
            val_losses=history["val_losses"],
            scaler=scaler,
            async_write=True
        )
        print(f"[train] checkpoint saved: {ckpt_path}")

    wait_for_pending_checkpoint()

    # * Artefact final *
    if config.save_final_model_state:
        save_final_model_state(model=model, final_model_path=final_model_path, device=device, config=config)
        print(f"[train] final model state saved: {final_model_path}")

    return history
