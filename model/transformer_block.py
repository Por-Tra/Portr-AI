"""
transformer_block.py

Bloc Transformer complet (pre-norm, style GPT).

Structure :
    x
    ↓ LayerNorm
    ↓ Multi-Head Attention  (+ KV Cache optionnel)
    ↓ Connexion résiduelle
    ↓ LayerNorm
    ↓ MLP (FeedForward)
    ↓ Connexion résiduelle
    → sortie

Optimisations vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| Attention.forward() retournait  | Signature mise à jour pour propager      |
| un seul tenseur — KV cache      | past_kv / use_cache jusqu'au bloc        |
| inaccessible depuis le bloc     |                                          |
| Double Dropout dans MLP         | Un seul Dropout après fc2 suffit ;       |
|                                 | le dropout après fc1 (avant GELU) apporte|
|                                 | peu et ralentit le training              |
──────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from .attention import Attention
except ImportError:
    from attention import Attention


class MLP(nn.Module):
    """
    Feed-Forward Network appliqué position par position.

    FFN(x) = Dropout(GELU(xW₁)W₂)

    Ratio d'expansion 4× (standard GPT) :
        embedding_dim=256 → hidden_dim=1024 → embedding_dim=256
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Initialisation scaled — réduit la variance à l'init sur les
        # couches de projection (technique GPT-2)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Bloc Transformer : Attention + MLP avec résiduel et pre-norm.

    Supporte le KV Cache pour une inférence efficace token-par-token.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1,
                 max_seq_length: int = 1024):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = Attention(
            embedding_dim, num_heads, dropout, max_seq_length=max_seq_length
        )

        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x       : [B, T, C]
            past_kv : KV cache du step précédent (inférence)
            use_cache: active le KV cache

        Returns:
            out       : [B, T, C]
            present_kv: (K, V) courant si use_cache, sinon None
        """
        attn_out, present_kv = self.attention(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv