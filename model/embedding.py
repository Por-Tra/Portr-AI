"""
Embedding.py

Couche d'embedding pour un modèle GPT.

Pipeline :
    tokens [batch, seq]
    ↓ nn.Embedding (token)       — gradients sparse : seuls les tokens vus
    ↓ nn.Embedding (position)      dans le batch sont mis à jour
    ↓ addition
    ↓ LayerNorm
    ↓ Dropout
    → [batch, seq, embedding_dim]

Optimisations vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| nn.Parameter pour token_emb :   | nn.Embedding → gradients sparse :        |
| gradient dense sur tout le      | seules les lignes des tokens du batch    |
| vocab à chaque step             | reçoivent un gradient → ×N plus rapide  |
|                                 | sur un grand vocabulaire (250k+)         |
| Pas de weight tying possible    | nn.Embedding expose .weight, compatible  |
|                                 | avec le weight tying dans GPT            |
──────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Token + Position embedding avec LayerNorm et Dropout.

    Utilise nn.Embedding (et non nn.Parameter) pour bénéficier des
    gradients sparse : sur un vocabulaire de 250k tokens, seule une
    infime fraction est vue par batch — mettre à jour les 250k lignes à
    chaque step est inutile et coûteux.
    """

    def __init__(self, vocab_size: int, seq_length: int, embedding_dim: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            vocab_size   : Nombre de tokens uniques dans le vocabulaire
            seq_length   : Longueur maximale des séquences
            embedding_dim: Dimension des vecteurs d'embedding
            dropout      : Taux de dropout
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # nn.Embedding → gradients sparse, compatible weight tying
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.position_emb = nn.Embedding(seq_length, embedding_dim)

        # Initialisation identique à GPT-2
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_emb.weight, mean=0.0, std=0.02)

        # Position ids pré-calculés (évite un arange à chaque forward)
        self.register_buffer(
            "position_ids",
            torch.arange(seq_length).unsqueeze(0),  # [1, seq_length]
            persistent=False,
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, seq_len] — indices des tokens

        Returns:
            [batch_size, seq_len, embedding_dim]
        """
        _, seq_len = tokens.shape

        token_vecs = self.token_emb(tokens)                         # [B, T, D]
        pos_vecs = self.position_emb(self.position_ids[:, :seq_len])  # [1, T, D]

        x = self.layer_norm(token_vecs + pos_vecs)
        return self.dropout(x)