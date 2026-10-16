# -*- coding: utf-8 -*-
"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : optimizer.py

description du script:
    Création de l'optimizer AdamW pour l'entraînement GPT.
"""


from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def create_adamw_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.01, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8):
    """ Crée un optimizer AdamW avec déduplication des paramètres (weight tying) et détection du kernel fused.

    Args:
        model (nn.Module): Modèle GPT pour lequel créer l'optimizer.
        lr (float, optional): Learning rate initial. Defaults to 3e-4.
        weight_decay (float, optional): Pénalité L2 sur les poids matriciels. Defaults to 0.01.
        betas (tuple[float, float], optional): Coefficients momentum AdamW (β1, β2). Defaults to (0.9, 0.95).
        eps (float, optional): Epsilon de stabilité numérique. Defaults to 1e-8.

    Returns:
        torch.optim.Optimizer: Instance de l'optimizer AdamW configuré pour le modèle.
    """

    #* 1. Dédupliquer les paramètres (weight tying) *
    #& output_head.weight et token_emb.weight pointent sur le même tenseur.
    #& Sans déduplication, ce tenseur apparaît dans deux groupes -> gradient
    #& appliqué deux fois et weight_decay aussi.
    seen_ids: set[int] = set()
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue  #& paramètre partagé déjà compté
        seen_ids.add(id(param))

        #& LayerNorm (poids 1D), biais et scalaires -> pas de weight decay.
        #& Le critère ndim suffit : les sous-chaînes "ln"/"norm" dans le nom
        #& attrapaient n'importe quel paramètre au nom mal choisi.
        if param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay}, 
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    #* 2. Détection robuste du kernel fused *
    #& inspect.signature() lit les paramètres réellement déclarés dans
    #& __init__, sans dépendre du contenu de la docstring.
    _adamw_sig = inspect.signature(torch.optim.AdamW)
    all_params = decay_params + no_decay_params
    params_on_cuda = bool(all_params) and all(p.is_cuda for p in all_params)
    use_fused = params_on_cuda and "fused" in _adamw_sig.parameters

    extra_kwargs: dict = {}
    if use_fused:
        extra_kwargs["fused"] = True
    else:
        #& foreach = boucle vectorisée sur CPU/GPU, plus rapide que la version
        #& scalaire par défaut même sans fused kernel
        extra_kwargs["foreach"] = True

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, **extra_kwargs)

    _log_optimizer_info(decay_params, no_decay_params, use_fused)
    return optimizer


def _log_optimizer_info(decay_params: list,no_decay_params: list,fused: bool):
    """ Log les informations sur les paramètres de l'optimizer pour le débogage et la vérification.

    Args:
        decay_params (list): paramètres soumis à weight decay (poids matriciels).
        no_decay_params (list): paramètres non soumis à weight decay (LayerNorm, biais, scalaires).
        fused (bool): Indique si le kernel fused est utilisé pour l'optimizer.
    """
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    total = n_decay + n_no_decay
    mode = "fused" if fused else "foreach"
    print(f"[optimizer] {total:,} params | decay={n_decay:,} | no_decay={n_no_decay:,} | mode={mode}")