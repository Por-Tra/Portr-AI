"""
Embedding.py

Ce fichier contient la classe Embedding, qui est responsable de la création des embeddings pour les tokens et les positions dans le modèle de langage. 
Les embeddings sont des représentations vectorielles denses qui capturent les relations sémantiques entre les tokens et les positions dans une séquence. 
La classe Embedding fournit des méthodes pour générer ces embeddings à partir des tokens d'entrée, ce qui est essentiel pour l'entraînement du modèle de langage.

L'embedding fonctionne en deux étapes principales :
1. Token Embedding : Cette étape convertit les tokens d'entrée en vecteurs denses en utilisant une matrice d'embedding. 
Chaque token est associé à un vecteur d'une dimension spécifiée (embedding_dim), qui capture les caractéristiques sémantiques du token.
2. Positionnal Embedding : Cette étape ajoute des informations de position aux embeddings des tokens. 
Comme les modèles de langage traitent des séquences de tokens, il est important de fournir des informations sur la position de chaque token dans la séquence.

"""

import torch
import torch.nn as nn




class Embedding(nn.Module):
    """
    1. Architecture réelle (simplifiée)

    Dans un GPT, l’entrée du modèle est construite comme ça :

    tokens
    [batch, seq]

    ↓ token embedding
    [batch, seq, dim]

    ↓ position embedding
    [batch, seq, dim]

    ↓ addition
    [batch, seq, dim]

    ↓ layer normalization  ← détail important
    [batch, seq, dim]

    ↓ dropout
    [batch, seq, dim]

    ↓ transformer

    Les deux éléments souvent oubliés :

    - LayerNorm
    - Dropout
    """

    def __init__(self, vocab_size, seq_length, embedding_dim=256):
        """
        Constructeur de la couche d'embedding.

        Args:
            vocab_size: Nombre de tokens uniques dans le vocabulaire
            seq_length: Longueur maximale des séquences
            embedding_dim: Dimension des vecteurs d'embedding
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # matrices
        self.token_emb = nn.Parameter(
            torch.randn(vocab_size, embedding_dim) * 0.02
        )

        self.position_emb = nn.Parameter(
            torch.randn(seq_length, embedding_dim) * 0.02
        )

        # normalisation + dropout
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens):
        """
        Pipeline complet d'embedding.

        Args:
            tokens: [batch_size, seq_length] - indices des tokens

        Returns:
            [batch_size, seq_length, embedding_dim]
        """
        batch_size, seq_len = tokens.shape
        
        # Token embedding
        token_vecs = self.token_emb[tokens]  # [batch_size, seq_len, embedding_dim]
        
        # Position embedding
        pos_vecs = self.position_emb[:seq_len]  # [seq_len, embedding_dim]
        pos_vecs = pos_vecs.unsqueeze(0)  # [1, seq_len, embedding_dim]
        
        # Addition
        x = token_vecs + pos_vecs
        
        # LayerNorm + Dropout
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x


       

    