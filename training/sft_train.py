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

File name : sft_train.py

description : Boucle d'entraînement supervisé pour les exemples instruction/réponse.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from training.loss import autoregressive_cross_entropy
from training.optimizer import create_adamw_optimizer
from training.train import TrainConfig, train_model, resolve_device


@dataclass
class SFTConfig(TrainConfig):
    """Configuration SFT avec un learning rate et des artefacts dédiés."""

    epochs: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    checkpoint_dir: str = "artifacts/sft/checkpoints"
    checkpoint_name: str = "sft_latest.pt"
    final_model_dir: str = "artifacts/sft/models"
    final_model_name: str = "sft_final_state.pt"
    resume_from_checkpoint: bool = True
    label_smoothing: float = 0.0
    z_loss_weight: float = 0.0
    benchmark_path: str = "artifacts/benchmarks/sft_training_benchmark.csv"
    unlikelihood_weight = 0.1   #& Poids de la loss unlikelihood (0 = désactivée)
    unlikelihood_ngram_size = 4  #& Taille des n-grammes à pénaliser
    unlikelihood_window = 64   #& Fenêtre de contexte pour la détection des répétitions
    unlikelihood_exclude_token_ids: tuple[int, ...] = ()    #& IDs de tokens à exclure de la pénalisation (ex: padding, prompt, etc.)
    tokenizer_fingerprint = None


def train_sft_model(model, dataloader, config = None, val_dataloader=None, optimizer = None):
    """Fine-tune un GPT sur des batches (input_ids, labels).

    Les positions dont le label vaut -100 sont ignorées par la loss. Le
    dataset SFT peut donc masquer le prompt et le padding sans traitement
    supplémentaire dans la boucle.

    Args:
        model (torch.nn.Module): Modèle GPT à fine-tuner.
        dataloader (torch.utils.data.DataLoader): Dataloader PyTorch pour l'entraînement.
        config (SFTConfig | None, optional): Configuration SFT. Defaults to None.
        val_dataloader (torch.utils.data.DataLoader, optional): Dataloader PyTorch pour la validation. Defaults to None.
        optimizer (torch.optim.Optimizer | None, optional): Optimiseur PyTorch. Defaults to None.

    Returns:
        [torch.nn.Module]: Modèle entraîné.
    """
    config = config or SFTConfig()
    if optimizer is None: model.to(resolve_device(config.device))

    optimizer = optimizer or create_adamw_optimizer(
        model,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    def loss_fn(logits, labels):
        return autoregressive_cross_entropy(
            logits,
            labels,
            ignore_index= -100,
            label_smoothing=config.label_smoothing,
            z_loss_weight=config.z_loss_weight
        )

    return train_model(
        model=model,
        dataloader=dataloader,
        optimizer= optimizer,
        loss_fn=loss_fn,
        config=config,
        val_dataloader=val_dataloader,
        eval_loss_fn=autoregressive_cross_entropy
    )


__all__ = ["SFTConfig", "train_sft_model"]