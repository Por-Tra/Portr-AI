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




class Embedding:
    """
    Schéma complet de l'ordre d'exe
        tokens
    [batch, seq]

    ↓ token embedding

    [batch, seq, dim]

    ↓ position embedding

    [batch, seq, dim]

    ↓ addition

    [batch, seq, dim]

    ↓ transformer
    """

    def __init__(self, vocab_size, embedding_dim, embedding_DIM = 256):
        """_summary_
        Constructeur de la classe Embedding, qui initialise les paramètres nécessaires pour créer une couche d'embedding.

        Args:
            vocab_size (_type_): _description_
            embedding_dim (_type_): _description_
            embedding_DIM (int, optional): _description_. Defaults to 256.
        """

        self.EMBEDDING_DIM = embedding_DIM

    def token_Embedding(self, tokens):
        # token_id -> vecteur
        pass

    def positionnal_Embedding(self, tokens):
        # position 0 -> vecteur
        # position 1 -> vecteur
        # position 2 -> vecteur

        # Puis
        # input = token_embedding + position_embedding
        
        pass

