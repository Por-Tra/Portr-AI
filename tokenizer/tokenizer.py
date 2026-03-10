"""Implémentation simple de Byte Pair Encoding (BPE).

Ce module gère les 2 phases :
1. Training : apprendre les fusions les plus fréquentes jusqu'à une taille cible.
2. Utilisation : tokeniser un nouveau texte en appliquant les fusions dans le même ordre.

faibles.
===============================================================================================================================
pseudo code BPE :
1. Initialiser le vocabulaire avec les caractères individuels + marqueur de fin de mot.
2. Compter la fréquence de toutes les paires adjacentes dans le corpus.
3. Trouver la paire la plus fréquente.
4. Si la fréquence est suffisante, fusionner cette paire dans tout le corpus et l'ajouter au vocabulaire.
5. Répéter à partir de l'étape 2 jusqu'à atteindre la taille de vocabulaire cible ou que les fréquences soient trop faibles.
===============================================================================================================================

Usage :
    python tokenizer.py
    (le training est rapide sur un petit corpus de test, mais peut être long sur de grands textes)
    Ensuite, le modèle est sauvegardé dans tokenizer/token/bpe_model.json et peut être chargé pour tokeniser de nouveaux textes :
    from tokenizer.tokenizer import Tokenizer
    tok = Tokenizer.load("tokenizer/token/bpe_model.json")
    tokens = tok.tokenize_text("votre texte à tokeniser ici")
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple
import re


DEFAULT_MODEL_PATH = "tokenizer/token/bpe_model.json"
SPACE_TOKEN = "▁"


class Tokenizer:
    """Tokeniseur BPE caractère -> sous-mots."""

    def __init__(self, end_of_word: str = "</w>"):
        self.end_of_word = end_of_word # Marqueur de fin de mot
        self.vocab: set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>"
        ]

        self.vocab.update(self.special_tokens)

        self.token_to_id = {}
        self.id_to_token = {}

    # ------------------------------------------------------------------
    # Découpage et traitement des données
    # ------------------------------------------------------------------
    
    def split_text(self, text: str) -> List[str]:
        """
        Découpe texte en mots + ponctuation
        tout en gardant les espaces.
        """
        pattern = r"\s+|[\w]+|[^\w\s]"
        return re.findall(pattern, text, re.UNICODE)

    def normalize_token(self, token: str) -> str:
        if token.isspace():
            return SPACE_TOKEN
        return token

    # ------------------------------------------------------------------
    # Préparation corpus
    # ------------------------------------------------------------------

    def prepare_word(self, word: str) -> List[str]:
        """Convertit un mot en symboles caractères + marqueur fin de mot."""
        if not word:
            return [self.end_of_word]
        return list(word) + [self.end_of_word]

    def prepare_corpus(self, texts: Iterable[str]) -> List[List[str]]:
        """Découpe un corpus texte en liste de mots représentés en symboles."""
        corpus_symbols: List[List[str]] = []

        for text in texts:
            for token in self.split_text(text):
                token = self.normalize_token(token)
                corpus_symbols.append(self.prepare_word(token))

        return corpus_symbols

    
    # ------------------------------------------------------------------
    # Cœur BPE
    # ------------------------------------------------------------------

    def count_pairs(self, corpus_symbols: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Compte la fréquence de toutes les paires adjacentes."""
        pair_counts: Dict[Tuple[str, str], int] = {}
        for symbols in corpus_symbols:
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    def most_frequent_pair(self, pair_counts: Dict[Tuple[str, str], int]) -> Optional[Tuple[str, str]]:
        """Retourne la paire la plus fréquente."""
        if not pair_counts:
            return None
        return max(pair_counts, key=pair_counts.get)

    def merge_pair(self, corpus_symbols: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Fusionne une paire donnée dans tout le corpus."""
        a, b = pair
        merged_token = a + b
        new_corpus: List[List[str]] = []

        for symbols in corpus_symbols:
            i = 0
            out: List[str] = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    out.append(merged_token)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            new_corpus.append(out)

        return new_corpus

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts: Iterable[str], target_vocab_size: int = 5000, min_pair_freq: int = 2, verbose: bool = False,) -> None:
        """Apprend les fusions BPE jusqu'à atteindre la taille de vocab cible.

        Args:
            texts: Corpus brut (liste de phrases/documents).
            target_vocab_size: Taille cible du vocabulaire final.
            min_pair_freq: Stop si la meilleure paire apparaît moins que ce seuil.
            verbose: Affiche la progression.
        """
        corpus_symbols = self.prepare_corpus(texts)
        self.vocab = set(sym for word in corpus_symbols for sym in word)
        self.merges = []
        self.merge_ranks = {}

        while len(self.vocab) < target_vocab_size:
            pair_counts = self.count_pairs(corpus_symbols)
            best_pair = self.most_frequent_pair(pair_counts)

            if best_pair is None:
                if verbose:
                    print("[BPE] Arrêt: plus de paires disponibles")
                break

            best_freq = pair_counts[best_pair]
            if best_freq < min_pair_freq:
                if verbose:
                    print(f"[BPE] Arrêt: fréquence max trop faible ({best_freq})")
                break

            corpus_symbols = self.merge_pair(corpus_symbols, best_pair)
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.add(merged_token)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1

            if verbose and len(self.merges) % 100 == 0:
                print(f"[BPE] merges={len(self.merges)} | vocab={len(self.vocab)} | best={best_pair} ({best_freq})")

        # après entrainement
        self.token_to_id = {tok:i for i,tok in enumerate(sorted(self.vocab))}
        self.id_to_token = {i:tok for tok,i in self.token_to_id.items()}

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """Applique les merges appris dans l'ordre d'apprentissage."""
        while True:

            pairs = [
                (symbols[i], symbols[i+1])
                for i in range(len(symbols)-1)
            ]

            ranked_pairs = [
                (self.merge_ranks[p], i)
                for i, p in enumerate(pairs)
                if p in self.merge_ranks
            ]

            if not ranked_pairs:
                break

            _, idx = min(ranked_pairs)

            a, b = symbols[idx], symbols[idx+1]
            merged = a + b

            symbols = (
                symbols[:idx] +
                [merged] +
                symbols[idx+2:]
            )

        return symbols

    def tokenize_word(self, word: str, strip_eow: bool = True) -> List[str]:
        """Tokenise un mot avec le modèle BPE appris."""
        symbols = self.prepare_word(word)
        tokens = self._apply_merges(symbols)
        if strip_eow:
            tokens = [t.replace(self.end_of_word, "") for t in tokens if t != self.end_of_word]
        return tokens

    def tokenize_text(self, text: str, strip_eow: bool = True) -> List[str]:
        """Tokenise un texte complet en concaténant les tokens de chaque mot."""
        output: List[str] = []

        for token in self.split_text(text):
            token = self.normalize_token(token)
            output.extend(self.tokenize_word(token, strip_eow=strip_eow))

        return output

    # ------------------------------------------------------------------
    # Sérialisation
    # ------------------------------------------------------------------

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        """Sauvegarde vocab + merges en JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "end_of_word": self.end_of_word,
            "vocab": sorted(self.vocab),
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        data = {
            "end_of_word": self.end_of_word,
            "vocab": sorted(self.vocab),
            "merges": [[a, b] for a, b in self.merges],
            "token_to_id": self.token_to_id
        }

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "Tokenizer":
        """Charge un modèle BPE sauvegardé."""

        
        try:

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            tok = cls(end_of_word=data.get("end_of_word", "</w>"))
            tok.vocab = set(data.get("vocab", []))
            tok.merges = [tuple(x) for x in data.get("merges", [])]
            tok.merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}

            tok.token_to_id = data.get("token_to_id", {})
            tok.id_to_token = {i:t for t,i in tok.token_to_id.items()}

            return tok
    
        except FileNotFoundError:
            print(f"[Tokenizer] Fichier de modèle introuvable : {path}")


    # ------------------------------------------------------------------
    # encoding/decoding (optionnel, pas demandé dans l'énoncé)
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_text(text)
        return [self.token_to_id.get(t, self.token_to_id["<UNK>"]) for t in tokens]