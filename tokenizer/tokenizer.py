"""
tokenizer.py

Tokenisateur de texte pour le projet de collecte de données web.
Ce module implémente un tokenisateur simple basé sur les espaces et la ponctuation.
Il est basé dur l'algorithme de tokenisation Byte Pair Encoding (BPE), utilisé par de nombreux modèles de langage comme GPT-2.
"""



class Tokenizer:
    
    def __init__(self):
        pass

    def encode(self, text: str) -> list[str]:
        pass

    def decode(self, tokens: list[str]) -> str:
        pass
