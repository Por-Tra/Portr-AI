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


def autoregressive_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100, label_smoothing: float = 0.0,
                                  z_loss_weight: float = 0.0) -> torch.Tensor:
    """
    Cross entropy autorégressif avec label smoothing et z-loss optionnels.

    Args:
        logits          : [batch, seq, vocab_size]
        targets         : [batch, seq]
        ignore_index    : token ignoré dans la loss (padding)
        label_smoothing : 0.1 = distribue 10% de la masse sur le reste du vocab.
                          Réduit l'overconfidence et améliore la généralisation.
                          Valeur recommandée : 0.1
        z_loss_weight   : Pénalise log(sum(exp(logits))) pour éviter que les
                          logits ne divergent vers ±inf (instabilité softmax).
                          Valeur recommandée : 1e-4. Mettre 0.0 pour désactiver.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected logits dim=3, got shape={tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets dim=2, got shape={tuple(targets.shape)}")

    bsz, seq_len, vocab_size = logits.shape
    if targets.shape != (bsz, seq_len):
        raise ValueError(f"targets shape mismatch: logits={tuple(logits.shape)}, targets={tuple(targets.shape)}")

    flat_logits = logits.view(-1, vocab_size)   # view évite une copie si contiguous
    flat_targets = targets.view(-1)

    # * Cross entropy + label smoothing *
    loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, label_smoothing=label_smoothing,)

    # * Z-loss (optionnel) *
    #& log(sum(exp(z)))² pénalise les logits qui s'éloignent de 0.
    #& Évite les saturations softmax qui bloquent les gradients.
    if z_loss_weight > 0.0:
        #& Masquer les positions padding pour ne pas les inclure dans z-loss
        mask = flat_targets != ignore_index
        if mask.any():
            log_z = torch.logsumexp(flat_logits[mask], dim=-1)
            z_loss = (log_z ** 2).mean()
            loss = loss + z_loss_weight * z_loss

    return loss


def perplexity_from_loss(loss_value: float) -> float:
    """Perplexité à partir d'une cross-entropy scalaire."""
    return float(torch.exp(torch.tensor(loss_value)).item())