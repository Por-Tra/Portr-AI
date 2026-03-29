"""
attention.py

Implémentation:
Q = Query
K = Key
V = Value

Formule de l'attention:
Attention(Q, K, V) = softmax(QKᵀ / √d) V
https://fr.wikipedia.org/wiki/Attention_(apprentissage_automatique)

La pipeline:
embedding
↓
Q, K, V projections
↓
score = Q @ Kᵀ
↓
masque causal (tokens ne voient pas le futur)
↓
softmax
↓
dropout
↓
pondération avec V
↓
concat heads + out_proj
↓
dropout + connexion résiduelle + LayerNorm
↓
output
"""

import torch
import torch.nn as nn
import numpy as np


class Attention(nn.Module):
    """
    Implémentation de l'attention multi-tête causale (style GPT).

    Les deux éléments souvent oubliés :
    - LayerNorm (appliquée AVANT l'attention, style "pre-norm")
    - Dropout (sur les poids d'attention ET sur la sortie)

    Les deux éléments souvent carrément absents :
    - Masque causal (les tokens ne voient pas le futur)
    - Connexion résiduelle (x + attention(x))
    - Projection de sortie (out_proj)
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
        self.q_proj    = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj    = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj    = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.layer_norm   = nn.LayerNorm(embedding_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout  = nn.Dropout(dropout)

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch_size, seq_length, embedding_dim]
        Returns:
            [batch_size, seq_length, embedding_dim]
        """
        B, T, _ = x.size()
        residual = x
 
        x = self._norm(x)
        Q, K, V = self._project_qkv(x)
        Q, K, V = self._split_heads(Q, K, V, B, T)
        x = self._attend(Q, K, V)
        x = self._merge_heads(x, B, T)
        x = self._out_project(x)
        x = self._residual(x, residual)
 
        return x
        

    def _norm(self, x):
        """
        Applique la normalisation de couche (LayerNorm) sur l'entrée x.

        Args:
            x (tensor): _description_
        """
        return self.layer_norm(x)
    
    def _project_qkv(self, x):
        """
        Projette l'entrée x en Q, K, V à l'aide de trois couches linéaires.

        Args:
            x (tensor): [batch_size, seq_length, embedding_dim]

        Returns:
            Q, K, V : chacun de shape [batch_size, seq_length, embedding_dim]
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        return Q, K, V
        
    def _split_heads(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, B: int, T: int,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Étape 3 — Découpe en heads.
        Divise l'embedding en num_heads sous-espaces indépendants.
        Chaque head apprend à "regarder" un aspect différent du contexte.
 
            [B, T, C] → [B, T, num_heads, head_dim] → [B, num_heads, T, head_dim]
        """
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
 
        return split(Q), split(K), split(V)
 
    def _attend(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,) -> torch.Tensor:
        """
        Étape 4 — Calcul de l'attention.
        C'est le cœur du mécanisme :
 
            scores       = Q @ Kᵀ / √d          → à quel point Q et K sont alignés
            masque causal                         → on cache le futur (-inf)
            attn_weights = softmax(scores)        → distribution de probabilité
            output       = attn_weights @ V       → somme pondérée des valeurs
        """
        T = Q.size(-2)
 
        # Scores bruts : similarité entre chaque paire (query_i, key_j)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [B, num_heads, T, T]
 
        # Masque causal : token i ne voit que les tokens 0..i (triangle inférieur)
        causal_mask = torch.tril(torch.ones(T, T, device=Q.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        scores = scores.masked_fill(~causal_mask, float("-inf"))
 
        # Softmax : convertit les scores en poids (somme = 1)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
 
        # Sortie : chaque token reçoit une somme pondérée des valeurs
        return torch.matmul(attn_weights, V)
        # [B, num_heads, T, head_dim]
 
    def _merge_heads(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Étape 5 — Recolle les heads.
        Inverse de _split_heads : reconstitue l'embedding complet.
 
            [B, num_heads, T, head_dim] → [B, T, num_heads, head_dim] → [B, T, C]
        """
        return x.transpose(1, 2).contiguous().view(B, T, self.embedding_dim)
 
    def _out_project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Étape 6 — Projection de sortie.
        Mélange les informations des différents heads en un seul vecteur,
        puis applique un dropout pour régulariser.
        """
        return self.out_dropout(self.out_proj(x))
 
    def _residual(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Étape 7 — Connexion résiduelle.
        Additionne la sortie de l'attention au x original (avant pre-norm).
        Permet à l'information de "court-circuiter" le bloc → gradients plus stables.
        """
        return residual + x

