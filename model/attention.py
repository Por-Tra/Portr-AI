"""
attention.py

Implémentation de l'attention multi-tête causale (style GPT).

Formule :
Attention(Q, K, V) = softmax(QKᵀ / √d) V

La pipeline :
1. Q, K, V projections -> [B, T, embedding_dim]
2. Split en heads -> [B, num_heads, T, head_dim]
3. Scores = Q @ Kᵀ / √d
4. Masque causal (tokens ne voient pas le futur)
5. Softmax -> distribution de probabilité
6. Pondération avec V
7. Merge heads + projection de sortie
8. Output -> [B, T, embedding_dim]

NOTE : Les connexions résiduelles et LayerNorm sont gérées
au niveau du TransformerBlock, pas ici !
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Implémentation pure du mécanisme d'attention multi-tête causale.
    
    Pas de LayerNorm ou résidual ici - juste le calcul d'attention !
    Le TransformerBlock gère les connexions résiduelles et normalisations.
    """

    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # Vérification que embedding_dim est divisible par num_heads
        assert embedding_dim % num_heads == 0, "embedding_dim doit être divisible par num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / √d

        # Projections linéaires pour Q, K, V
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch_size, seq_length, embedding_dim]
        Returns:
            [batch_size, seq_length, embedding_dim]
        """
        B, T, _ = x.size()

        # Étape 1 : Projections Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Étape 2 : Split en heads
        Q, K, V = self._split_heads(Q, K, V, B, T)

        # Étape 3-6 : Attention
        x = self._attend(Q, K, V, T)

        # Étape 7 : Merge heads
        x = self._merge_heads(x, B, T)

        # Étape 8 : Projection finale
        x = self.out_dropout(self.out_proj(x))

        return x

    def _split_heads(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, B: int, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Divise l'embedding en num_heads sous-espaces indépendants.
        
        [B, T, C] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        """
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        return split(Q), split(K), split(V)

    def _attend(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, T: int) -> torch.Tensor:
        """
        Calcul du mécanisme d'attention.
        
        1. Scores = Q @ Kᵀ / √d
        2. Masque causal (tokens ne voient pas le futur)
        3. Softmax
        4. Pondération avec V
        """
        # Scores bruts : similarité entre chaque paire (query_i, key_j)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [B, num_heads, T, T]

        # Masque causal : token i ne voit que les tokens 0..i (triangle inférieur)
        causal_mask = torch.tril(torch.ones(T, T, device=Q.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        scores = scores.masked_fill(~causal_mask, -1e9)

        # Softmax : convertit les scores en poids (somme = 1)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Sortie : chaque token reçoit une somme pondérée des valeurs
        return torch.matmul(attn_weights, V)
        # [B, num_heads, T, head_dim]

    def _merge_heads(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Recolle les heads.
        
        [B, num_heads, T, head_dim] -> [B, T, num_heads, head_dim] -> [B, T, C]
        """
        return x.transpose(1, 2).contiguous().view(B, T, self.embedding_dim)


