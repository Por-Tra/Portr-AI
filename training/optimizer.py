"""
optimizer.py

Création de l'optimizer AdamW pour l'entraînement GPT.

Corrections vs version précédente :
------------------------------------------------------------------------------
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| Détection de fused via          | inspect.signature() -> accès direct aux  |
| __doc__ : fragile sur builds    | paramètres déclarés, ne dépend pas du   |
| custom ou si la docstring       | contenu de la docstring                  |
| change                          |                                          |
------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn


def create_adamw_optimizer(
    model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.01, betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8
) -> torch.optim.Optimizer:
    """
    Crée un optimizer AdamW avec :
    - déduplication des paramètres partagés (weight tying)
    - séparation decay / no-decay (LayerNorm, biais exclus du weight decay)
    - kernel fused sur CUDA (PyTorch ≥ 2.0) -> ~1.3x plus rapide
    - foreach comme fallback si fused indisponible

    Args:
        model        : Modèle GPT.
        lr           : Learning rate initial.
        weight_decay : Pénalité L2 sur les poids matriciels.
        betas        : Coefficients momentum AdamW (β1, β2).
        eps          : Epsilon de stabilité numérique.

    Returns:
        Optimizer AdamW configuré.
    """

    # * 1. Dédupliquer les paramètres (weight tying) *
    #& output_head.weight et token_emb.weight pointent sur le même tenseur.
    #& Sans déduplication, ce tenseur apparaît dans deux groupes -> gradient
    #& appliqué deux fois et weight_decay aussi.
    seen_ids: set[int] = set()
    decay_params:    list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue  #& paramètre partagé déjà compté
        seen_ids.add(id(param))

        #& LayerNorm, biais et scalaires -> pas de weight decay
        if (
            param.ndim <= 1
            or name.endswith("bias")
            or "norm" in name.lower()
            or "ln" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # * 2. Détection robuste du kernel fused *
    #& inspect.signature() lit les paramètres réellement déclarés dans
    #& __init__, sans dépendre du contenu de la docstring.
    _adamw_sig = inspect.signature(torch.optim.AdamW)
    use_fused = (torch.cuda.is_available() and "fused" in _adamw_sig.parameters)

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


def _log_optimizer_info(decay_params: list,no_decay_params: list,fused: bool) -> None:
    """Affiche un résumé de la configuration de l'optimizer."""
    n_decay    = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    total      = n_decay + n_no_decay
    mode       = "fused" if fused else "foreach"
    print(f"[optimizer] {total:,} params | decay={n_decay:,} | no_decay={n_no_decay:,} | mode={mode}")