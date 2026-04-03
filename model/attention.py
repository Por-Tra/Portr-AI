"""
attention.py

Implémentation de l'attention multi-tête causale (style GPT).

Formule :
    Attention(Q, K, V) = softmax(QKᵀ / √d) V

Pipeline :
    1. Projection QKV fusionnée → [B, T, 3·C]
    2. Split en heads           → [B, H, T, head_dim]
    3. Concat avec KV cache     → [B, H, T_full, head_dim]
    4. Flash Attention (sdpa) ou fallback manuel
    5. Merge heads              → [B, T, C]
    6. Projection de sortie     → [B, T, C]

Corrections vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| is_causal=True invalide avec    | Quand past_kv is not None (decode loop), |
| KV cache : pendant le decode,   | on passe is_causal=False car Q[T=1] doit |
| T=1 et T_full > 1. sdpa        | voir tout le cache sans masque causal.   |
| construisait un masque [1×1]    | Pendant le training et le prefill        |
| au lieu de [1×T_full], laissant | (past_kv=None), is_causal=True reste     |
| passer des positions futures    | valide et optimal.                       |
──────────────────────────────────────────────────────────────────────────────

NOTE : Connexions résiduelles et LayerNorm gérées dans TransformerBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Attention(nn.Module):
    """
    Attention multi-tête causale avec Flash Attention et KV Cache.

    - Flash Attention via F.scaled_dot_product_attention :
        évite de matérialiser la matrice [B, H, T, T] → 2-4× moins de VRAM.

    - KV Cache (use_cache=True) :
        pendant le decode, seul le token courant est projeté (T=1) ;
        K et V des steps précédents sont lus depuis le cache → O(1) par step.

    - Buffer causal_mask :
        fallback pour PyTorch < 2.0 (sans sdpa), alloué une seule fois.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
    ):
        super().__init__()

        assert embedding_dim % num_heads == 0, \
            "embedding_dim doit être divisible par num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.scale         = self.head_dim ** -0.5
        self.dropout_p     = dropout

        # Projections Q, K, V fusionnées (1 appel kernel au lieu de 3)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)

        # Masque causal pré-calculé (fallback sans sdpa)
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(max_seq_length, max_seq_length, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0),  # [1, 1, T, T]
            persistent=False,
        )

        # Détection Flash Attention (PyTorch ≥ 2.0)
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x        : [B, T, C]  — T=1 pendant le decode avec KV cache
            past_kv  : (K_cache, V_cache) de forme [B, H, T_past, head_dim]
            use_cache: si True, retourne le KV courant pour le prochain appel

        Returns:
            out       : [B, T, C]
            present_kv: (K, V) ou None
        """
        B, T, C = x.shape

        # ── Projection QKV ────────────────────────────────────────────────
        qkv = self.qkv_proj(x)                      # [B, T, 3C]
        Q, K, V = qkv.split(self.embedding_dim, dim=-1)

        # ── Split en heads [B, H, T, head_dim] ───────────────────────────
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ── KV Cache : concat avec le passé ──────────────────────────────
        is_decoding = past_kv is not None
        if is_decoding:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        present_kv = (K, V) if use_cache else None
        T_full = K.shape[2]  # longueur totale après concat

        # ── Attention ─────────────────────────────────────────────────────
        if self._use_flash:
            # Règle is_causal :
            #
            #   Training / prefill (past_kv=None, T == T_full) :
            #     is_causal=True → masque triangulaire inférieur standard.
            #     Correct et optimal.
            #
            #   Decode (past_kv non-None, T=1, T_full > 1) :
            #     is_causal=True demanderait à sdpa un masque [T×T_full]
            #     où seule la ligne Q[0] serait active. PyTorch applique
            #     en réalité un masque [1×1] qui bloque toutes les positions
            #     du cache sauf la dernière → les K/V passés sont ignorés.
            #     is_causal=False est correct : Q[t] peut légitimement
            #     voir tous les tokens passés (ils sont antérieurs).
            is_causal = not is_decoding

            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Fallback manuel avec buffer pré-calculé
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            mask   = self.causal_mask[:, :, :T, :T_full]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_out = torch.matmul(torch.softmax(scores, dim=-1), V)

        # ── Merge heads [B, T, C] ─────────────────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # ── Projection de sortie ──────────────────────────────────────────
        out = self.out_dropout(self.out_proj(attn_out))

        return out, present_kv