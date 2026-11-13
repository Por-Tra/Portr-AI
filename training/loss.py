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

File name : loss.py

description du script:
    Fonctions de calcul de la loss pour l'entraînement GPT.
"""




from __future__ import annotations

import torch
import torch.nn.functional as F
import math
from numbers import Real


def validate_unlikelihood_parameters(weight, ngram_size, window):
    """ Vérifie les paramètres pour la loss unlikelihood (UL).

    Args:
        weight (float): Le poids de la perte unlikelihood.
        ngram_size (int): La taille des n-grammes à considérer.
        window (int): La taille de la fenêtre pour la recherche des n-grammes.

    Raises:
        ValueError: Si le poids n'est pas un nombre fini strictement positif.
        ValueError: Si la taille des n-grammes n'est pas un entier >= 2.
        ValueError: Si la taille de la fenêtre n'est pas un entier >= taille des n-grammes.
    """
    if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("unlikelihood weight doit etre un nombre fini strictement positif (> 0)")
    if not isinstance(ngram_size, int) or isinstance(ngram_size, bool) or ngram_size < 2:
        raise ValueError("unlikelihood ngram_size doit etre un entier >= 2")
    if not isinstance(window, int) or isinstance(window, bool) or window < ngram_size:
        raise ValueError("unlikelihood window doit etre un entier >= ngram_size")


def repetition_unlikelihood(logits, targets, *, ngram_size = 4, window = 64, ignore_index = -100, 
                            exclude_token_ids: tuple[int, ...] = ()):
    """ Calcul de la loss unlikelihood pour pénaliser les répétitions de n-grammes dans les séquences générées.

    Args:
        logits (torch.Tensor): Logits du modèle de forme (batch_size, seq_length, vocab_size)
        targets (torch.Tensor): Cibles de forme (batch_size, seq_length)
        ngram_size (int, optional): La taille des n-grammes à considérer. Defaults to 4.
        window (int, optional): La taille de la fenêtre pour la recherche des n-grammes. Defaults to 64.
        ignore_index (int, optional): L'index à ignorer dans les cibles. Defaults to -100.
        exclude_token_ids (tuple[int, ...], optional): Les identifiants de tokens à exclure. Defaults to ().

    Raises:
        ValueError: Si le poids n'est pas un nombre fini strictement positif.
        ValueError: Si la taille des n-grammes n'est pas un entier >= 2.
        ValueError: Si la taille de la fenêtre n'est pas un entier >= taille des n-grammes.
        ValueError: Si les identifiants de tokens à exclure ne sont pas dans le vocabulaire.

    Returns:
        torch.Tensor: La loss unlikelihood.
    """
    if not isinstance(ngram_size, int) or isinstance(ngram_size, bool) or ngram_size < 2:
        raise ValueError("unlikelihood ngram_size doit etre un entier >= 2")
    if not isinstance(window, int) or isinstance(window, bool) or window < ngram_size:
        raise ValueError("unlikelihood window doit etre un entier >= ngram_size")
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("unlikelihood attend logits [B,T,V] et targets [B,T]")
    batch, length, vocab = logits.shape
    if any(token_id < 0 or token_id >= vocab for token_id in exclude_token_ids):
        raise ValueError("unlikelihood exclude_token_ids hors vocabulaire")
    if length <= ngram_size:
        return logits[..., 0].sum() * 0.0


    # & Masque des positions supervisées (non padding) et valides (non exclues)
    supervised = targets != ignore_index
    valid = supervised.clone()
    for token_id in exclude_token_ids:  #& Exclure les tokens spéciaux du calcul de la loss unlikelihood
        valid &= targets != token_id

    #& Segmentation des séquences pour identifier les répétitions de n-grammes
    segments = (~valid).long().cumsum(dim=1)
    position = torch.arange(length, device=targets.device)[:, None]
    distance = torch.arange(1, min(window, length) + 1, device=targets.device)[None, :]
    previous = position - distance
    safe_previous = previous.clamp_min(0)
    candidates = targets[:, safe_previous]
    matches = (previous >= ngram_size - 1)[None, :, :] & valid[:, :, None]
    matches = matches & (segments[:, safe_previous] == segments[:, :, None])

    # & Vérification des répétitions de n-grammes
    for offset in range(1, ngram_size):
        current_index = (position[:, 0] - offset).clamp_min(0)
        old_index = (previous - offset).clamp_min(0)

        #& Récupération des préfixes actuels et anciens
        current_prefix = targets[:, current_index]
        old_prefix = targets[:, old_index]
        matches &= current_prefix[:, :, None] == old_prefix
        matches &= valid[:, current_index, None] & valid[:, old_index]
    matches &= candidates != targets[:, :, None]  # A correct repeat is never negative.
    matches &= candidates != ignore_index
    for token_id in exclude_token_ids:  #& Exclure les tokens spéciaux du calcul de la loss unlikelihood
        matches &= candidates != token_id

    #& Sort candidates to remove duplicates and keep only unique negative tokens
    candidates = torch.where(matches, candidates, vocab).sort(dim=-1).values
    unique = candidates != vocab
    unique[:, :, 1:] &= candidates[:, :, 1:] != candidates[:, :, :-1]
    active = unique.any(dim=-1)
    selected_logits = logits[active]

    #& Vérification du type de tenseur pour éviter les erreurs de type lors du calcul des probabilités
    if selected_logits.dtype in (torch.float16, torch.bfloat16):
        selected_logits = selected_logits.float()

    #& Récupération des IDs des tokens négatifs et de leur masque
    negative_ids = candidates[active].clamp_max(vocab - 1)
    negative_mask = unique[active]
    log_probs = F.log_softmax(selected_logits, dim=-1).gather(1, negative_ids)

    #& Calcul des probabilités et pénalités pour les tokens négatifs
    probabilities = torch.where(negative_mask, log_probs.exp(), 0.0)
    penalties = -torch.log1p(-probabilities.clamp_max(1.0 - 1e-6))
    return penalties.sum() / supervised.sum().clamp_min(1)


def training_loss_with_unlikelihood(logits, targets, *, base_loss_fn, weight = 0.1,
                                     ngram_size = 4, window = 64,
                                       ignore_index = -100, exclude_token_ids: tuple[int, ...] = ()):
    """ Calcul de la loss d'entraînement combinant la cross-entropy et la loss unlikelihood
      pour pénaliser les répétitions de n-grammes.

    Args:
        logits (torch.Tensor): Logits du modèle de forme (batch_size, seq_length, vocab_size)
        targets (torch.Tensor): Cibles de forme (batch_size, seq_length)
        base_loss_fn ((torch.Tensor, torch.Tensor) -> torch.Tensor): Fonction de calcul de la loss de base (cross-entropy).
        weight (float, optional): Poids de la loss unlikelihood. Defaults to 0.1.
        ngram_size (int, optional): Taille des n-grammes à pénaliser. Defaults to 4.
        window (int, optional): Taille de la fenêtre glissante pour détecter les répétitions. Defaults to 64.
        ignore_index (int, optional): Index des tokens à ignorer dans la loss (padding). Defaults to -100.
        exclude_token_ids (tuple[int, ...], optional): IDs des tokens à exclure du calcul de la loss unlikelihood. Defaults to ().

    Returns:
        torch.Tensor: Valeur scalaire de la loss combinée.
    """
    validate_unlikelihood_parameters(weight, ngram_size, window)
    base_loss = base_loss_fn(logits, targets)
    return base_loss + weight * repetition_unlikelihood(
        logits, targets, ngram_size=ngram_size, window=window,
        ignore_index=ignore_index, exclude_token_ids=exclude_token_ids
    )


def autoregressive_cross_entropy(logits, targets, ignore_index = -100, 
                                 label_smoothing = 0.0, z_loss_weight = 0.0):
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