"""
optimizer.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def create_adamw_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    AdamW avec :
    - exclusion correcte du weight tying (token_emb partagé avec output_head)
    - fused kernel sur CUDA (PyTorch ≥ 2.0) → ~1.3× plus rapide
    - foreach comme fallback si fused indisponible
    """

    #* ── 1. Dédupliquer les paramètres (weight tying) ──────────────────────
    #& output_head.weight et token_emb.weight pointent sur le même tenseur.
    #& Sans déduplication, ce tenseur apparaît dans deux groupes → le gradient
    #& est appliqué deux fois et le weight_decay aussi.
    seen_ids: set[int] = set()
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue                    # ← ignore les paramètres partagés
        seen_ids.add(id(param))

        if (
            param.ndim <= 1                     # scalaires, biais
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

    #* ── 2. Fused AdamW sur CUDA ───────────────────────────────────────────
    #& Le kernel fused exécute la mise à jour en un seul pass GPU au lieu
    #& de plusieurs opérations élément par élément → moins de round-trips
    #& mémoire, ~1.3× plus rapide sur GPU.
    use_fused = (
        torch.cuda.is_available()
        and "fused" in torch.optim.AdamW.__init__.__doc__  # PyTorch ≥ 2.0
    )

    extra_kwargs: dict = {}
    if use_fused:
        extra_kwargs["fused"] = True
    else:
        #& foreach = boucle vectorisée sur CPU/GPU, plus rapide que la version
        #& scalaire par défaut même sans fused kernel
        extra_kwargs["foreach"] = True

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, **extra_kwargs)

    _log_optimizer_info(model, decay_params, no_decay_params, use_fused)
    return optimizer


def _log_optimizer_info(model: nn.Module, decay_params: list, no_decay_params: list, fused: bool) -> None:
    """
    optimise le calcul

    Args:
        model (nn.Module): _description_
        decay_params (list): _description_
        no_decay_params (list): _description_
        fused (bool): _description_
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    mode = "fused" if fused else "foreach"
    print(f"[optimizer] {total:,} params | decay={n_decay:,} | no_decay={n_no_decay:,} | mode={mode}")