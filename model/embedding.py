# -*- coding: utf-8 -*-

"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : Embedding.py

description du script:
    Couche d'embedding pour un modèle GPT.

    Pipeline :
        tokens [batch, seq]
        | nn.Embedding (token)       — gradients sparse : seuls les tokens vus
        | nn.Embedding (position)      dans le batch sont mis à jour
        | addition
        | Dropout
        -> [batch, seq, embedding_dim]
"""

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Token + Position embedding avec Dropout.

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

        # nn.Embedding -> gradients sparse, compatible weight tying
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

        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, seq_len] — indices des tokens
            pos_offset: decalage de position absolue pour l'inference
                token-par-token (KV cache).

        Returns:
            [batch_size, seq_len, embedding_dim]
        """
        _, seq_len = tokens.shape
        end_pos = pos_offset + seq_len
        if end_pos > self.seq_length:
            raise ValueError(
                f"Position out of range: pos_offset={pos_offset}, seq_len={seq_len}, "
                f"max_seq_length={self.seq_length}"
            )

        token_vecs = self.token_emb(tokens)                         # [B, T, D]
        positions = self.position_ids[:, pos_offset:end_pos]
        pos_vecs = self.position_emb(positions)  # [1, T, D]

        x = token_vecs + pos_vecs
        return self.dropout(x)