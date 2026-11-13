# -*- coding: utf-8 -*-
"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @Lucas Contreras Hodapp

File name : train.py

description du script:
    Script d'entraînement du modèle GPT sur le dataset préparé.

"""


from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
import math
import os
import threading
import time
from functools import partial

import torch
from training.loss import training_loss_with_unlikelihood, validate_unlikelihood_parameters


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
    benchmark_path: str = "artifacts/benchmarks/training_benchmark.csv"

    #* torch.compile (nécessite Triton : Linux, ou triton-windows sous Windows)
    compile_model: bool = False
    unlikelihood_weight = 0.1   #& Poids de la loss unlikelihood (0 = désactivée)
    unlikelihood_ngram_size= 4  #& Taille des n-grammes à pénaliser
    unlikelihood_window: int = 64   #& Fenêtre de contexte pour la détection des répétitions
    unlikelihood_exclude_token_ids: tuple[int, ...] = ()    #& IDs de tokens à exclure de 
                                                            #& la pénalisation (ex: padding, prompt, etc.)
    tokenizer_fingerprint = None
    data_identity: dict = field(default_factory=dict)

    def __post_init__(self):
        _unlikelihood_settings(self)
        if self.epochs <= 0 or self.grad_accum_steps <= 0 or self.log_interval <= 0:
            raise ValueError("epochs, grad_accum_steps et log_interval doivent etre positifs")
        if self.eval_max_batches is not None and self.eval_max_batches <= 0:
            raise ValueError("eval_max_batches doit etre positif ou None")
        if self.warmup_steps < 0 or (self.warmup_ratio is not None and not 0 <= self.warmup_ratio <= 1):
            raise ValueError("Warmup invalide")
        if not 0 <= self.lr_decay_factor <= 1:
            raise ValueError("lr_decay_factor doit etre compris entre 0 et 1")


def _unlikelihood_settings(config):
    """Versioned objective metadata for new checkpoints and final artifacts."""
    weight = config.unlikelihood_weight
    validate_unlikelihood_parameters(weight, config.unlikelihood_ngram_size, config.unlikelihood_window)
    return {
        "version": 1, "weight": float(weight),
        "ngram_size": config.unlikelihood_ngram_size,
        "window": config.unlikelihood_window,
        "exclude_token_ids": sorted(set(config.unlikelihood_exclude_token_ids)),
    }


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

    Raises:
        ValueError: Si le type de données demandé n'est pas "auto", "bf16" ou "fp16".
        RuntimeError: Si bf16 est demandé mais non supporté par le GPU.

    Returns:
        torch.dtype: Le type de données résolu.
    """
    if requested not in {"auto", "bf16", "fp16"}:
        raise ValueError("amp_dtype doit etre auto, bf16 ou fp16")
    if requested == "bf16":

        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 demande mais non supporte par le GPU")
        
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

_pending_save = None   #& threading.Thread
_pending_save_error = None    #& Exception


def _rng_state():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng_state(state):
    """ Restaure l'état aléatoire de PyTorch (CPU et GPU) à partir d'un dictionnaire.

    Args:
        state (dict): Dictionnaire contenant l'état aléatoire à restaurer.

    Raises:
        RuntimeError: Si le nombre de GPU est incompatible avec l'état aléatoire sauvegardé.
    """
    torch.set_rng_state(state["cpu"])
    if state.get("cuda") and torch.cuda.is_available():

        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Nombre de GPU incompatible avec l'etat aleatoire sauvegarde.")
        
        torch.cuda.set_rng_state_all(state["cuda"])


def _architecture_contract(model):
    return {
        "vocab_size": getattr(model, "vocab_size", None),
        "seq_length": getattr(model, "seq_length", None),
        "embedding_dim": getattr(model, "embedding_dim", None),
        "num_blocks": getattr(model, "num_blocks", None),
        "num_heads": model.blocks[0].attention.num_heads if hasattr(model, "blocks") and len(model.blocks) else None,
        "dropout": model.embedding.dropout.p if hasattr(model, "embedding") else None
    }


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


def _atomic_save(payload, path):
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
    """Attend la fin de l'écriture asynchrone en cours (si existante).

    Raises:
        RuntimeError: Si l'écriture asynchrone a échoué.
    """
    global _pending_save, _pending_save_error
    if _pending_save is not None:
        _pending_save.join()
        _pending_save = None
    if _pending_save_error is not None:
        error, _pending_save_error = _pending_save_error, None
        raise RuntimeError("Echec de sauvegarde du checkpoint asynchrone.") from error


def _async_save(payload: dict, path: Path):
    """ Sauvegarde asynchrone d'un checkpoint dans un thread séparé.

    Args:
        payload (dict): Dictionnaire contenant les informations à sauvegarder.
        path (Path): Chemin du fichier de checkpoint final.
    """
    global _pending_save
    wait_for_pending_checkpoint()

    def _write():
        global _pending_save_error
        try:
            _atomic_save(payload, path)
        except Exception as exc:
            _pending_save_error = exc

    _pending_save = threading.Thread(target=_write, daemon=False)
    _pending_save.start()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, batch_in_epoch: int,step: int,
    avg_loss: float, checkpoint_path: Path, total_steps: int,
      epoch_losses: list[float] | None = None, val_losses: list[float] | None = None,
        scaler: torch.amp.GradScaler | None = None, async_write: bool = True,
        unlikelihood_settings: dict | None = None, run_contract: dict | None = None):
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
    payload["rng_state"] = _rng_state()
    if run_contract is not None:
        payload["run_contract"] = run_contract
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if unlikelihood_settings is not None:
        payload["unlikelihood_settings"] = unlikelihood_settings

    payload = _to_cpu_snapshot(payload)

    if async_write:
        _async_save(payload, checkpoint_path)
    else:
        wait_for_pending_checkpoint()
        _atomic_save(payload, checkpoint_path)


def load_checkpoint_if_available(model, optimizer, scheduler, 
                                 checkpoint_path, resume, 
                                 scaler = None, target_epochs = None, expected_unlikelihood = None,
                                 expected_run_contract = None):
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
    if expected_unlikelihood is not None:
        saved = payload.get("unlikelihood_settings")
        validate_unlikelihood_parameters(expected_unlikelihood["weight"], expected_unlikelihood["ngram_size"], expected_unlikelihood["window"])
        if saved != expected_unlikelihood:
            raise RuntimeError("Objectif unlikelihood incompatible avec le checkpoint. Demarrer un nouveau run pour changer ses reglages.")
    effective_epoch = int(payload.get("epoch", 0))
    if expected_run_contract and int(payload.get("batch_in_epoch", 0)) >= expected_run_contract["steps_per_epoch"]:
        effective_epoch += 1
    if target_epochs is not None and effective_epoch >= target_epochs:
        raise RuntimeError(f"Entrainement deja termine: checkpoint epoch={payload.get('epoch')} \n pour target_epochs={target_epochs}. Utiliser un nouveau run pour \n repartir du modele de base (option 10 pour SFT), ou definir \n explicitement une nouvelle cible et son schedule pour continuer.")
    if expected_run_contract is not None and payload.get("run_contract") != expected_run_contract:
        raise RuntimeError("Contrat de reprise incompatible ou absent (donnees, batch, accumulation ou schedule modifies). Demarrer un nouveau run.")
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

    if "rng_state" in payload:  #& restaurer l'état aléatoire pour une reprise exacte
        _restore_rng_state(payload["rng_state"])

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
        "tokenizer_fingerprint": config.tokenizer_fingerprint,
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
            ) if hasattr(model, "embedding") else 0.0
        },
        "config": {
            "unlikelihood_settings": _unlikelihood_settings(config),
            "epochs":          int(config.epochs),
            "grad_clip":       float(config.grad_clip),
            "warmup_steps":    int(config.warmup_steps),
            "lr_decay_factor": float(config.lr_decay_factor),
            "use_amp":         bool(config.use_amp),
            "grad_accum_steps": int(config.grad_accum_steps)
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
    if max_steps_per_epoch <= 0:
        raise ValueError("max_steps_per_epoch doit etre positif")
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


def _validate_token_batch(model, x, y):
    if x.ndim != 2 or y.shape != x.shape or x.dtype not in (torch.int32, torch.int64) or y.dtype != torch.int64:
        raise ValueError("Batch attendu: input_ids entiers [B,T], labels int64 de meme forme")
    vocab = getattr(model, "vocab_size", None)
    if vocab is not None and ((x < 0).any() or (x >= vocab).any() or ((y != -100) & ((y < 0) | (y >= vocab))).any()):
        raise ValueError("IDs du batch hors vocabulaire: verifier caches et tokenizer")


def _optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler,
                     scaler: torch.amp.GradScaler, grad_clip: float, grad_normalizer: float = 1.0):
    """ Effectue un step de l'optimizer avec support pour l'AMP (Automatic Mixed Precision)
      et le clipping des gradients.

    Args:
        model (torch.nn.Module): Modèle à entraîner.
        optimizer (torch.optim.Optimizer): optimizer à utiliser pour la mise à jour des poids.
        scheduler (_type_): scheduler de learning rate à mettre à jour après le step.
        scaler (torch.amp.GradScaler): Scaler pour l'AMP, utilisé pour la mise à l'échelle des gradients.
        grad_clip (float): Valeur maximale pour le clipping des gradients. Si <= 0, aucun clipping n'est appliqué.

    Returns:
        tuple[torch.Tensor, bool]: Norme des gradients et indicateur de mise à jour effective.
    """
    scaler.unscale_(optimizer)
    for param in model.parameters():
        if param.grad is not None:
            param.grad.mul_(grad_normalizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), grad_clip if grad_clip > 0 else float("inf"),
        error_if_nonfinite=not scaler.is_enabled()
    )
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    did_update = scaler.get_scale() >= previous_scale
    if did_update:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm, did_update


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

    Raises:
        ValueError: Si max_batches est défini et inférieur ou égal à 0.
        ValueError: Si aucun token supervisé n'est trouvé dans l'ensemble de validation.

    Returns:
        float: La loss moyenne sur l'ensemble de validation.
    """
    was_training = model.training
    rng_before_eval = _rng_state()
    model.eval()
    total = torch.zeros((), device=device)
    count = 0
    if max_batches is not None and max_batches <= 0:
        model.train(was_training)
        raise ValueError("max_batches doit etre positif ou None")
    try:
        for batch_idx, (x, y) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            supervised = int((y != -100).sum().item())
            if supervised == 0:
                continue
            _validate_token_batch(model, x, y)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                batch_loss = loss_fn(model(x), y).float()
                if not torch.isfinite(batch_loss):
                    raise ValueError("Loss de validation non finie")
                total += batch_loss * supervised
            count += supervised
    finally:
        model.train(was_training)
        _restore_rng_state(rng_before_eval)
    if count == 0:
        raise ValueError("Validation sans token supervise.")
    return float(total.item()) / count


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
#* Benchmark
#* ---------------------------------------------------------------------------

def training_benchmark(epoch, batch,step, loss, LR, grad_norm, elapsed_seconds=None, tokens_per_second=None, val_loss=None, event="train", path="artifacts/benchmarks/training_benchmark.csv",):
    """ Enregistre les métriques d'entraînement dans un fichier CSV pour le suivi des performances.

    Args:
        epoch (_type_): _description_
        batch (_type_): _description_
        step (_type_): _description_
        loss (_type_): _description_
        LR (_type_): _description_
        grad_norm (_type_): _description_
        elapsed_seconds (_type_, optional): _description_. Defaults to None.
        tokens_per_second (_type_, optional): _description_. Defaults to None.
        path (str, optional): _description_. Defaults to "artifacts/benchmarks/training_benchmark.csv".
    """
    benchmark_path = Path(path)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "epoch",
        "batch",
        "step",
        "event",
        "loss",
        "val_loss",
        "learning_rate",
        "grad_norm",
        "elapsed_seconds",
        "tokens_per_second"
    ]

    should_migrate = False
    migrated_rows = []
    if benchmark_path.exists() and benchmark_path.stat().st_size > 0:
        with benchmark_path.open("r", newline="", encoding="utf-8") as source:

            reader = csv.DictReader(source)
            old_fieldnames = reader.fieldnames or []

            if set(fieldnames) - set(old_fieldnames):
                for row in reader:
                    row.setdefault("event", "train")
                    row.setdefault("val_loss", "")
                    migrated_rows.append(row)
                should_migrate = True

    if should_migrate:
        temporary_path = benchmark_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(migrated_rows)
        os.replace(temporary_path, benchmark_path)

    needs_header = not benchmark_path.exists() or benchmark_path.stat().st_size == 0

    with benchmark_path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()

        writer.writerow({
            "timestamp": time.time(),
            "epoch": int(epoch),
            "batch": int(batch),
            "step": int(step),
            "event": event,
            "loss": "" if loss is None else float(loss),
            "val_loss": "" if val_loss is None else float(val_loss),
            "learning_rate": float(LR),
            "grad_norm": float(grad_norm),
            "elapsed_seconds": "" if elapsed_seconds is None else float(elapsed_seconds),
            "tokens_per_second": "" if tokens_per_second is None else float(tokens_per_second)
        })

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
    config.__post_init__()
    ul_settings = _unlikelihood_settings(config)
    device = resolve_device(config.device)
    _apply_cpu_thread_limits(config)

    model.to(device)
    model.train()

    #& Le module compilé ne sert qu'au forward ; on sauvegarde toujours `model`
    #& (le state_dict d'un module compilé a des clés préfixées "_orig_mod.").
    forward_model = torch.compile(model) if config.compile_model else model
    eval_loss_fn = eval_loss_fn or loss_fn
    loss_fn = partial(
        training_loss_with_unlikelihood, base_loss_fn=loss_fn,
        weight=config.unlikelihood_weight, ngram_size=config.unlikelihood_ngram_size,
        window=config.unlikelihood_window,
        exclude_token_ids=config.unlikelihood_exclude_token_ids
    )
    print(f"[train] unlikelihood obligatoire: weight={config.unlikelihood_weight} ngram={config.unlikelihood_ngram_size} window={config.unlikelihood_window}; validation sans UL")

    accum = max(1, int(config.grad_accum_steps))
    steps_per_epoch = _resolve_steps_per_epoch(dataloader, config.max_steps_per_epoch)
    if steps_per_epoch <= 0: raise ValueError("Dataloader d'entrainement vide")
    #& Le scheduler avance une fois par optimizer step, pas par micro-batch
    optim_steps_per_epoch = math.ceil(steps_per_epoch / accum)
    total_steps = optim_steps_per_epoch * config.epochs

    warmup_steps = (int(total_steps * config.warmup_ratio) if config.warmup_ratio is not None else int(config.warmup_steps))
    config.warmup_steps = warmup_steps
    run_contract = {
        "version": 1, "normalization": "supervised_tokens", "epochs": config.epochs,
        "batch_size": getattr(dataloader, "batch_size", None), "grad_accum_steps": accum,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "warmup_steps": warmup_steps, "lr_decay_factor": config.lr_decay_factor,
        "seed": config.seed, "data_identity": config.data_identity,
        "tokenizer_fingerprint": config.tokenizer_fingerprint,
        "architecture": _architecture_contract(model),
        "dataset_length": len(dataloader.dataset) if hasattr(dataloader, "dataset") else None,
        "sampler": {"class": type(getattr(dataloader, "sampler", None)).__name__,
                    "seed": getattr(getattr(dataloader, "sampler", None), "seed", None),
                    "replacement": getattr(getattr(dataloader, "sampler", None), "replacement", None)},
        "optimizer": {"class": type(optimizer).__module__ + "." + type(optimizer).__name__,
                      "groups": [{key: group.get("initial_lr", group["lr"]) if key == "lr" else group[key]
                                  for key in ("lr", "weight_decay", "betas", "eps") if key in group}
                                 for group in optimizer.param_groups]}
    }

    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(device, config.amp_dtype)
    run_contract["precision"] = {"use_amp": use_amp, "dtype": str(amp_dtype) if use_amp else "float32"}
    #& GradScaler uniquement utile en fp16 (bf16 a la même plage que fp32)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp and amp_dtype == torch.float16)

    print(
        f"[train] device={device} | amp={'off' if not use_amp else str(amp_dtype).replace('torch.', '')} "
        f"| micro-batches/epoch={steps_per_epoch} | optimizer steps/epoch={optim_steps_per_epoch} "
        f"| total_steps={total_steps} | warmup={warmup_steps}"
    )

    ckpt_path = Path(config.checkpoint_dir) / config.checkpoint_name
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
    torch.manual_seed(config.seed)
    start_epoch, resume_batch, global_step, restored_losses, restored_val = load_checkpoint_if_available(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_path=ckpt_path,
        resume=config.resume_from_checkpoint,
        scaler=scaler,
        target_epochs=config.epochs,
        expected_unlikelihood=ul_settings,
        expected_run_contract=run_contract
    )
    history["epoch_losses"].extend(restored_losses)
    history["val_losses"].extend(restored_val)

    last_train_loss = None
    last_learning_rate = 0.0
    last_grad_norm = 0.0

    def _run_eval(tag: str): #& fonction interne pour évaluation périodique
        nonlocal last_train_loss, last_learning_rate, last_grad_norm
        if val_dataloader is None:
            return
        val_loss = evaluate(model, val_dataloader, eval_loss_fn, device, use_amp, amp_dtype, config.eval_max_batches)
        history["val_losses"].append(val_loss)
        ppl = math.exp(min(val_loss, 50.0))
        training_benchmark(
            epoch=epoch + 1,
            batch=0,
            step=global_step,
            loss=last_train_loss,
            val_loss=val_loss,
            LR=last_learning_rate,
            grad_norm=last_grad_norm,
            event="validation",
            path=config.benchmark_path,
        )
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
        running_tokens = 0
        steps = 0
        grad_norm = None
        #& Accumulateur GPU pour le log (évite .item() à chaque step)
        log_loss_accum = torch.zeros((), device=device)
        log_steps = 0
        log_tokens = 0
        log_input_tokens = 0
        group_tokens = 0
        group_reference = 1
        start_t = time.time()
        last_log_t = start_t

        optimizer.zero_grad(set_to_none=True)

        # DataLoader's base_seed must not consume the restored dropout RNG.
        iterator_rng = torch.get_rng_state()
        data_iter = iter(dataloader)
        torch.set_rng_state(iterator_rng)
        if skip and not sampler_skips:
            #& Fallback : sampler non reproductible -> on consomme les batches
            for _ in range(skip):
                next(data_iter)

        for batch_idx, (x, y) in enumerate(data_iter, start=skip):
            if batch_idx >= steps_per_epoch:
                break

            supervised_tokens = int((y != -100).sum().item())
            if supervised_tokens == 0:
                raise ValueError("Batch d'entrainement sans token supervise")
            _validate_token_batch(model, x, y)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            #& Le dernier groupe d'accumulation de l'epoch peut être incomplet :
            #& on normalise par sa taille réelle et on fait quand même le step.
            group_start = (batch_idx // accum) * accum
            group_size = min(accum, steps_per_epoch - group_start)
            is_last_accum = (batch_idx + 1 - group_start) == group_size
            if batch_idx == group_start:
                group_tokens = 0
                group_reference = max(group_size * x.numel(), 1)
            group_tokens += supervised_tokens

            # * Forward + backward *
            loss = _run_forward_loss(
                model=forward_model,
                loss_fn=loss_fn,
                x=x,
                y=y,
                accum_divisor=1,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )
            scaler.scale(loss * (supervised_tokens / group_reference)).backward()

            #& Pas de .item() ici : accumulation sur GPU, sync uniquement au log
            log_loss_accum += loss.detach() * supervised_tokens
            log_steps += 1
            log_tokens += supervised_tokens
            log_input_tokens += x.numel()
            steps += 1

            # * Optimizer step (tous les grad_accum_steps batches) *
            if is_last_accum:
                grad_norm, did_update = _optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    grad_clip=config.grad_clip,
                    grad_normalizer=group_reference / group_tokens
                )
                global_step += int(did_update)

                if did_update and config.checkpoint_every_steps > 0 and global_step % config.checkpoint_every_steps == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        batch_in_epoch=batch_idx + 1,
                        step=global_step,
                        avg_loss=(running_loss + float(log_loss_accum.item())) / max(running_tokens + log_tokens, 1),
                        checkpoint_path=ckpt_path,
                        total_steps=total_steps,
                        epoch_losses=history["epoch_losses"],
                        val_losses=history["val_losses"],
                        scaler=scaler,
                        async_write=True,
                        unlikelihood_settings=ul_settings,
                        run_contract=run_contract
                    )
                    print(f"[train] periodic checkpoint queued (step={global_step}): {ckpt_path}")

                if did_update and config.eval_interval > 0 and global_step % config.eval_interval == 0:
                    _run_eval("periodic")

            # * Log (sync GPU une seule fois par log_interval) *
            if (batch_idx + 1) % config.log_interval == 0:
                log_sum = float(log_loss_accum.item())
                running_loss += log_sum
                running_tokens += log_tokens
                avg_log_loss = log_sum / max(log_tokens, 1)
                logged_steps = log_steps
                log_elapsed = max(time.time() - last_log_t, 1e-9)
                last_log_t = time.time()
                log_loss_accum.zero_()
                log_steps = 0

                current_lr = scheduler.get_last_lr()[0]
                gn = float(grad_norm.item()) if grad_norm is not None else 0.0
                tokens_per_second = log_input_tokens / log_elapsed
                log_tokens = 0
                log_input_tokens = 0
                last_train_loss = avg_log_loss
                last_learning_rate = current_lr
                last_grad_norm = gn
                training_benchmark(
                    epoch=epoch + 1,
                    batch=batch_idx + 1,
                    step=global_step,
                    loss=avg_log_loss,
                    LR=current_lr,
                    grad_norm=gn,
                    elapsed_seconds=log_elapsed,
                    tokens_per_second=tokens_per_second,
                    path=config.benchmark_path,
                )
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
            running_tokens += log_tokens

        avg_loss = running_loss / max(running_tokens, 1)
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
            async_write=True,
            unlikelihood_settings=ul_settings,
            run_contract=run_contract
        )
        print(f"[train] checkpoint queued: {ckpt_path}")

    wait_for_pending_checkpoint()

    # * Artefact final *
    if config.save_final_model_state:
        save_final_model_state(model=model, final_model_path=final_model_path, device=device, config=config)
        print(f"[train] final model state saved: {final_model_path}")

    return history
