# -*- coding: utf-8 -*-
"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : loss.py

description du script:
    Fonctions de calcul de la loss pour l'entraînement GPT.
"""




from __future__ import annotations

import torch
import torch.nn.functional as F


def autoregressive_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100, label_smoothing: float = 0.0, z_loss_weight: float = 0.0):
    """ Calcul de la cross-entropy pour un modèle autoregressif (GPT).

    Args:
        logits (torch.Tensor): Logits du modèle de forme (batch_size, seq_length, vocab_size)
        targets (torch.Tensor): Cibles de forme (batch_size, seq_length)
        ignore_index (int, optional): Index des tokens à ignorer dans la loss (padding). Defaults to -100.
        label_smoothing (float, optional): Paramètre de label smoothing pour réduire l'overconfidence. Defaults to 0.0.
        z_loss_weight (float, optional): Pénalité sur la Z-loss pour éviter les saturations softmax. Defaults to 0.0.

    Raises:
        ValueError: Gestion des erreurs de paramètrage
        ValueError: Gestion des erreurs de paramètrage
        ValueError: Gestion des erreurs de paramètrage

    Returns:
        torch.Tensor: Valeur scalaire de la loss (cross-entropy + z-loss si applicable)
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits dim=3, got shape={tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets dim=2, got shape={tuple(targets.shape)}")

    bsz, seq_len, vocab_size = logits.shape
    if targets.shape != (bsz, seq_len):
        raise ValueError(f"targets shape mismatch: logits={tuple(logits.shape)}, targets={tuple(targets.shape)}")

    flat_logits = logits.reshape(-1, vocab_size)   #& reshape = view si contiguous, sinon copie
    flat_targets = targets.reshape(-1)

    # * Cross entropy + label smoothing *
    loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, label_smoothing=label_smoothing,)

    # * Z-loss (optionnel) *
    #& log(sum(exp(z)))² pénalise les logits qui s'éloignent de 0.
    #& Évite les saturations softmax qui bloquent les gradients.
    if z_loss_weight > 0.0:
        #& Masquer les positions padding pour ne pas les inclure dans z-loss
        mask = flat_targets != ignore_index
        if mask.any():
            #& Calcul en fp32 : sous autocast les logits peuvent être en fp16/bf16
            log_z = torch.logsumexp(flat_logits[mask].float(), dim=-1)
            z_loss = (log_z ** 2).mean()
            loss = loss + z_loss_weight * z_loss

    return loss


def perplexity_from_loss(loss_value: float):
    """ Calcul de la perplexité à partir de la loss (cross-entropy).
    La perplexité est définie comme exp(loss) pour la cross-entropy.

    Args:
        loss_value (float): Valeur scalaire de la loss (cross-entropy)

    Returns:
        float: Description de la perplexité calculée.
    """
    return float(torch.exp(torch.tensor(loss_value)).item())