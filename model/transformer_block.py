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

Corrections vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| MLP.__init__ initialisait fc1   | Init manuelle supprimée de MLP :         |
| et fc2 avec std=0.02 AVANT que  | GPT._apply_scaled_init() est le seul     |
| GPT._apply_scaled_init() ne     | responsable de l'init des projections    |
| ré-initialise fc2 avec le bon   | résiduelles (fc2 avec std scalé,         |
| std scalé → double init, fc2    | fc1 avec std=0.02). Ordre garanti,       |
| recevait l'init de GPT en       | pas de collision.                        |
| dernier mais fc1 gardait 0.02   |                                          |
──────────────────────────────────────────────────────────────────────────────

NOTE : Les connexions résiduelles et LayerNorm sont gérées ici.
       L'attention multi-tête est définie dans attention.py.
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

    FFN(x) = Dropout(fc2(GELU(fc1(x))))

    Ratio d'expansion 4× (standard GPT) :
        embedding_dim → 4 × embedding_dim → embedding_dim

    Init : GPT._apply_scaled_init() se charge de tout après la construction
    du modèle complet. Ne pas initialiser ici pour éviter les conflits d'ordre.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.fc1     = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.fc2     = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # ⚠️  Pas d'init manuelle ici.
        # GPT._apply_scaled_init() applique :
        #   - fc1 : std=0.02  (projection d'entrée, pas résiduelle)
        #   - fc2 : std=0.02/√(2N)  (projection résiduelle, scaled GPT-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Bloc Transformer : Attention + MLP avec résiduel et pre-norm.

    Supporte le KV Cache pour une inférence efficace token-par-token.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_length: int = 1024,
    ):
        super().__init__()

        self.ln1       = nn.LayerNorm(embedding_dim)
        self.attention = Attention(
            embedding_dim,
            num_heads,
            dropout,
            max_seq_length=max_seq_length,
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
            x        : [B, T, C]
            past_kv  : KV cache du step précédent (inférence uniquement)
            use_cache: si True, retourne le KV cache courant

        Returns:
            out       : [B, T, C]
            present_kv: (K, V) courant si use_cache=True, sinon None
        """
        # Pre-norm + attention + résiduel
        attn_out, present_kv = self.attention(
            self.ln1(x), past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_out

        # Pre-norm + MLP + résiduel
        x = x + self.mlp(self.ln2(x))

        return x, present_kv