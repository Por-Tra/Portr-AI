# -*- coding: utf-8 -*-

"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : transformer_block.py

description du script:
    Bloc Transformer complet (pre-norm, style GPT).

    Structure :
        x
        | LayerNorm
        | Multi-Head Attention  (+ KV Cache optionnel)
        | Connexion résiduelle
        | LayerNorm
        | MLP (FeedForward)
        | Connexion résiduelle
        V 
      sortie

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

    Ratio d'expansion 4x (standard GPT) :
        embedding_dim -> 4 x embedding_dim -> embedding_dim

    Init : GPT._apply_scaled_init() se charge de tout après la construction
    du modèle complet. Ne pas initialiser ici pour éviter les conflits d'ordre.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.fc1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        #! /!\  Pas d'init manuelle ici.
        #! GPT._apply_scaled_init() applique :
        #!   - fc1 : std=0.02  (projection d'entrée, pas résiduelle)
        #!   - fc2 : std=0.02/√(2N)  (projection résiduelle, scaled GPT-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Bloc Transformer : Attention + MLP avec résiduel et pre-norm.

    Supporte le KV Cache pour une inférence efficace token-par-token.
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1, max_seq_length: int = 1024):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = Attention(
            embedding_dim,
            num_heads,
            dropout,
            max_seq_length=max_seq_length
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, dropout)

    def forward(self, x: torch.Tensor, past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """_summary_

        Args:
            x (torch.Tensor): [B, T, C]  — T=1 pendant le decode avec KV cache
            past_kv (Optional[Tuple[torch.Tensor, torch.Tensor]], optional): KV cache du step précédent (inférence uniquement). Defaults to None.
            use_cache (bool, optional): si True, retourne le KV cache courant. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: [B, T, C] et le KV cache courant si use_cache=True, sinon None
        """
        
        #& Pre-norm + attention + résiduel
        attn_out, present_kv = self.attention(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out

        #& Pre-norm + MLP + résiduel
        x = x + self.mlp(self.ln2(x))

        return x, present_kv