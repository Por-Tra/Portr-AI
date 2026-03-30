"""
attention.py

Implémentation de l'attention multi-tête causale (style GPT).

Formule :
Attention(Q, K, V) = softmax(QKᵀ / √d) V

La pipeline :
1. Q, K, V projections -> [B, T, embedding_dim]
2. Split en heads -> [B, num_heads, T, head_dim]
3. Flash Attention (scaled_dot_product_attention) avec masque causal
4. Merge heads + projection de sortie
5. Output -> [B, T, embedding_dim]

Optimisations vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| Matrice scores [B,H,T,T] en RAM | Flash Attention (sdpa) — pas de          |
|                                 | matérialisation explicite                |
| Masque causal recréé à chaque   | register_buffer() — alloué une seule     |
| forward (torch.tril + ones)     | fois, réutilisé comme fallback           |
| Dropout géré manuellement       | sdpa le gère en interne (fused)          |
| KV recalculés entièrement à     | KV Cache optionnel — seul le dernier     |
| chaque étape de génération      | token est projeté pendant l'inférence    |
──────────────────────────────────────────────────────────────────────────────

NOTE : Les connexions résiduelles et LayerNorm sont gérées
au niveau du TransformerBlock, pas ici !
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Attention(nn.Module):
    """
    Attention multi-tête causale avec Flash Attention et KV Cache.

    - Flash Attention via F.scaled_dot_product_attention :
        évite de matérialiser la matrice [B, H, T, T] → 2-4× plus rapide
        et consomme beaucoup moins de VRAM sur de longues séquences.

    - KV Cache (use_cache=True) :
        pendant la génération, on ne calcule K et V que pour le nouveau token
        et on concatène avec le cache passé → O(1) par étape au lieu de O(T).

    - Buffer causal_mask :
        fallback si sdpa n'est pas disponible (PyTorch < 2.0),
        alloué une seule fois dans __init__.
    """

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 max_seq_length: int = 1024):
        super().__init__()

        assert embedding_dim % num_heads == 0, \
            "embedding_dim doit être divisible par num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout

        # Projections Q, K, V fusionnées en une seule matrice pour réduire
        # les appels kernel et améliorer la localité mémoire.
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)

        # Masque causal pré-calculé (fallback si sdpa indisponible)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool))
            .unsqueeze(0).unsqueeze(0),  # [1, 1, T, T]
            persistent=False,
        )

        # Détection Flash Attention (PyTorch >= 2.0)
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x        : [B, T, C]  — T = 1 pendant la génération avec KV cache
            past_kv  : (K_cache, V_cache) de forme [B, H, T_past, head_dim]
            use_cache: si True, retourne (K, V) courant pour le prochain appel

        Returns:
            out      : [B, T, C]
            present_kv: (K, V) ou None
        """
        B, T, C = x.shape

        # ── Projection QKV fusionnée ──────────────────────────────────────
        # [B, T, 3C] → trois tenseurs [B, T, C]
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.split(self.embedding_dim, dim=-1)

        # ── Split en heads [B, H, T, head_dim] ───────────────────────────
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ── KV Cache : concaténer avec le passé ───────────────────────────
        if past_kv is not None:
            past_K, past_V = past_kv
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        present_kv = (K, V) if use_cache else None
        T_full = K.shape[2]  # longueur totale après concat

        # ── Attention ─────────────────────────────────────────────────────
        if self._use_flash:
            # Flash Attention : O(T) VRAM, masque causal géré en interne.
            # is_causal=True suppose que Q voit toujours les T_full premières
            # positions ; correct pendant le training. Pendant la génération
            # (T=1), is_causal=True est également valide car Q[0] peut
            # voir tout le cache passé.
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Fallback manuel avec le buffer pré-calculé
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            mask = self.causal_mask[:, :, :T, :T_full]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_out = torch.matmul(torch.softmax(scores, dim=-1), V)

        # ── Merge heads [B, T, C] ─────────────────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        # ── Projection de sortie ──────────────────────────────────────────
        out = self.out_dropout(self.out_proj(attn_out))

        return out, present_kv