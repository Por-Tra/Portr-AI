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
        """_summary_
        Constructeur de la classe Embedding, qui initialise les paramètres nécessaires pour créer une couche d'embedding.

        Args:
            vocab_size (_type_): _description_
            embedding_dim (_type_): _description_
            embedding_DIM (int, optional): _description_. Defaults to 256.
        """

        super().__init__()

        self.embedding_dim = embedding_dim
        self.seq_length = seq_length

        # matrices
        self.token_embedding = nn.Parameter(
            torch.randn(vocab_size, embedding_dim) * 0.02
        )

        self.position_embedding = nn.Parameter(
            torch.randn(seq_length, embedding_dim) * 0.02
        )

        # normalisation + dropout
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, tokens):
        """
        Initialise la pipeline d'embedding.

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
        """
        token_vec = self.token_embedding(tokens)
        pos_vec = self.positional_embedding(tokens.shape[1], tokens.device)

        x = token_vec + pos_vec
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x
    

    def token_embedding(self, tokens):
        """


        Args:
            tokens (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.token_embedding[tokens]

    def positional_embedding(self, seq_len, device):
        """_summary_

        Args:
            seq_len (_type_): _description_
            device (_type_): _description_

        Returns:
            _type_: _description_
        """
        positions = torch.arange(seq_len, device=device)
        return self.position_embedding[positions].unsqueeze(0)


       

    