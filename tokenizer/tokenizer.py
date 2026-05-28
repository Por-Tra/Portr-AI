"""Implémentation optimisée de Byte Pair Encoding (BPE).


Version 1.1
Optimisations clés vs version naïve :
-----------------------------------------------------------------------------
| Problème original          | Solution appliquée                           |
|----------------------------|----------------------------------------------|
| count_pairs() O(N) x V     | Comptage incrémental — seules les paires     |
|                            | voisines du token fusionné sont mises à jour |
| merge_pair() O(N) x V      | word_freq dict : on ne stocke chaque forme   |
|                            | de mot qu'une seule fois (x sa fréquence)    |
| most_frequent_pair() O(P)  | Heap max avec lazy-deletion -> O(log P)      |
-----------------------------------------------------------------------------

Complexité :
    Naïve   : O(V x N)          N = nb total tokens dans corpus
    Optimisée: O(V x W x log P) W = nb mots uniques (W << N en général)

Et oui une complexité en log ça c'est beau (⌐■_■)
Pas moi qui ai fais l'optimisation :-D

Usage :
    python tokenizer.py
    from tokenizer.tokenizer import Tokenizer
    tok = Tokenizer.load("tokenizer/token/bpe_model.json")
    tokens = tok.tokenize_text("votre texte à tokeniser ici")

Version 1.2
Limiter le corpus d'entraînement aux mots pour éviter les redondances
massives (ex: "the" 10 000 fois).
Optimisation de l'encodage du corpus en utilisant des fonctions locales
et des structures de données plus efficaces.
Amélioration -> 10-20it/s -> 50-+1000it/s sur un corpus de 50 000 lignes.

Version 1.3
_apply_merges : scan unique sans listes intermédiaires + merge en place
               via indices (évite 2 allocations list par itération).
encode_corpus : cache word -> [ids] + array('i') pour le buffer de sortie.
"""

from __future__ import annotations

import heapq
import json
import os
import re
from array import array
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

DEFAULT_MODEL_PATH = "tokenizer/token/bpe_model.json"
SPACE_TOKEN = "▁"

DEFAULT_CORPUS_LIMIT = 1_000


class Tokenizer:
    """Tokeniseur BPE caractère -> sous-mots (implémentation optimisée)."""

    def __init__(self, end_of_word: str = "</w>"):
        self.end_of_word = end_of_word
        self.vocab: set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        self.special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.vocab.update(self.special_tokens)

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    #* ------------------------------------------------------------------
    #* Découpage et normalisation
    #* ------------------------------------------------------------------

    def split_text(self, text: str) -> List[str]:
        """Découpe texte en mots + ponctuation en conservant les espaces."""
        return re.findall(r"\s+|[\w]+|[^\w\s]", text, re.UNICODE)

    def normalize_token(self, token: str) -> str:
        return SPACE_TOKEN if token.isspace() else token

    #* ------------------------------------------------------------------
    #* Préparation corpus
    #* ------------------------------------------------------------------

    def prepare_word(self, word: str) -> List[str]:
        """Convertit un mot en symboles caractères + marqueur fin de mot."""
        if not word:
            return [self.end_of_word]
        return list(word) + [self.end_of_word]

    def _build_word_freq(self, texts: Iterable[str], limit: int = 0) -> Dict[Tuple[str, ...], int]:
        """
        Construit un dictionnaire { séquence_symboles -> fréquence }.

        Au lieu de stocker chaque occurrence du mot "the" 10 000 fois,
        on stocke  ('t','h','e','</w>') -> 10000.
        Cela réduit drastiquement le coût des itérations suivantes.
        """
        word_freq: Dict[Tuple[str, ...], int] = defaultdict(int)
        for i, text in enumerate(tqdm(texts, desc="Preparing corpus"), start=1):
            if limit and i > limit:
                break
            for token in self.split_text(text):
                token = self.normalize_token(token)
                key = tuple(self.prepare_word(token))
                word_freq[key] += 1
        return word_freq

    #* ------------------------------------------------------------------
    #* Comptage de paires (helpers internes)
    #* ------------------------------------------------------------------

    @staticmethod
    def _pairs_in_word(word: Tuple[str, ...]) -> List[Tuple[str, str]]:
        """Trouve toutes les paires de symboles adjacents dans un mot donné."""
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    def _count_all_pairs(self, word_freq: Dict[Tuple[str, ...], int]) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], set]]:
        """
        Comptage initial complet.
        Retourne :
          pair_counts  : paire -> fréquence agrégée
          pair_to_words: paire -> ensemble des mots qui la contiennent
        """
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[str, str], set] = defaultdict(set)

        for word, freq in word_freq.items():
            for pair in self._pairs_in_word(word):
                pair_counts[pair] += freq
                pair_to_words[pair].add(word)

        return pair_counts, pair_to_words

    #* ------------------------------------------------------------------
    #* Training (version optimisée)
    #* ------------------------------------------------------------------

    def train(self, texts: Iterable[str], target_vocab_size: int = 5000, min_pair_freq: int = 2, verbose: bool = False, corpus_limit: int = 0) -> None:
        """
        Apprend les fusions BPE.

        Optimisations actives :
           word_freq dict   -> évite de re-parcourir les doublons de mots
           mise à jour incrémentale des pair_counts après chaque merge
           heap max (lazy-deletion) -> meilleure paire en O(log P)
        """
        # * 1. Construction du dictionnaire de fréquences *
        word_freq = self._build_word_freq(texts, limit=corpus_limit)

        # * 2. Vocabulaire initial (caractères uniques) *
        self.vocab = set(self.special_tokens)
        for word in word_freq:
            self.vocab.update(word)
        self.merges = []
        self.merge_ranks = {}

        # * 3. Comptage initial de toutes les paires *
        pair_counts, pair_to_words = self._count_all_pairs(word_freq)

        # * 4. Heap max (on stocke des tuples (-fréquence, paire)) *
        heap: List[Tuple[int, Tuple[str, str]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        pbar = tqdm(total=target_vocab_size, desc="Training BPE")

        while len(self.vocab) < target_vocab_size:

            # * Trouver la meilleure paire via le heap *
            best_pair: Optional[Tuple[str, str]] = None
            best_freq = 0

            while heap:
                neg_cnt, candidate = heapq.heappop(heap)
                if pair_counts.get(candidate, 0) == -neg_cnt:
                    best_pair = candidate
                    best_freq = -neg_cnt
                    break

            if best_pair is None or best_freq < min_pair_freq:
                if verbose:
                    reason = (
                        "plus de paires"
                        if best_pair is None
                        else f"fréquence trop faible ({best_freq})"
                    )
                    print(f"\n[BPE] Arrêt : {reason}")
                break

            # * Fusionner *
            a, b = best_pair
            merged = a + b
            self.vocab.add(merged)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            pbar.update(1)

            affected_words = list(pair_to_words.pop(best_pair, set()))

            for old_word in affected_words:
                freq = word_freq.get(old_word)
                if freq is None:
                    continue

                for pair in self._pairs_in_word(old_word):
                    cnt = pair_counts.get(pair, 0) - freq
                    if cnt <= 0:
                        pair_counts.pop(pair, None)
                        pair_to_words.pop(pair, None)
                    else:
                        pair_counts[pair] = cnt
                        pair_to_words[pair].discard(old_word)

                new_word_list: List[str] = []
                i = 0
                while i < len(old_word):
                    if (
                        i < len(old_word) - 1
                        and old_word[i] == a
                        and old_word[i + 1] == b
                    ):
                        new_word_list.append(merged)
                        i += 2
                    else:
                        new_word_list.append(old_word[i])
                        i += 1
                new_word = tuple(new_word_list)

                del word_freq[old_word]
                word_freq[new_word] = word_freq.get(new_word, 0) + freq

                for pair in self._pairs_in_word(new_word):
                    new_cnt = pair_counts.get(pair, 0) + freq
                    pair_counts[pair] = new_cnt
                    pair_to_words[pair].add(new_word)
                    heapq.heappush(heap, (-new_cnt, pair))

            # TODO: ASUP
            # Afficher les merges
            # if verbose and len(self.merges) % 100 == 0:
                # print(f"[BPE] merges={len(self.merges)} | vocab={len(self.vocab)} | best={best_pair} ({best_freq})")

        pbar.close()

        # * Finalisation *
        self.token_to_id = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    #* ------------------------------------------------------------------
    #* Tokenisation
    #* ------------------------------------------------------------------

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """
        Applique les merges appris à une séquence de symboles.

        Optimisation vs version précédente :
        -----------------------------------------------------------------
        Avant : à chaque itération, on construisait deux listes
                intermédiaires (pairs + ranked) de taille O(L), soit
                2 allocations heap par merge.

        Après : un seul scan linéaire avec variables locales.
                - merge_ranks.get() en accès direct (dict local)
                - best_rank / best_idx mis à jour in-place
                - une seule allocation pour le nouveau symbols
                -> ~2x moins d'allocations pour les mots courts/moyens
        -----------------------------------------------------------------

        Args:
            symbols: ex. ['t', 'h', 'e', '</w>']

        Returns:
            ex. ['the</w>'] après application des merges
        """
        merge_ranks = self.merge_ranks  #& référence locale (évite self. lookup)

        while len(symbols) >= 2:
            best_rank = -1          #& -1 = "rien trouvé"
            best_idx  = -1

            #& Scan unique : on cherche la paire de plus bas rang (= priorité max)
            for i in range(len(symbols) - 1):
                rank = merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank == -1 or rank < best_rank):
                    best_rank = rank
                    best_idx  = i

            if best_idx == -1:      #& plus aucune paire connue
                break

            #& Fusion : on reconstruit symbols autour de best_idx
            #& (une seule nouvelle liste, pas de liste 'pairs' intermédiaire)
            merged = symbols[best_idx] + symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2:]

        return symbols

    def tokenize_word(self, word: str, strip_eow: bool = True) -> List[str]:
        """
        Tokenise un mot donné en appliquant les merges appris.

        Args:
            word     : Le mot à tokeniser.
            strip_eow: Si True, retire le marqueur de fin de mot des tokens.

        Returns:
            Liste de tokens.
        """
        symbols = self.prepare_word(word)
        tokens = self._apply_merges(symbols)
        if strip_eow:
            tokens = [
                t.replace(self.end_of_word, "")
                for t in tokens
                if t != self.end_of_word
            ]
        return tokens

    def tokenize_text(self, text: str, strip_eow: bool = True) -> List[str]:
        """
        Tokenise un texte complet.

        Args:
            text     : Le texte à tokeniser.
            strip_eow: Si True, retire le marqueur de fin de mot.

        Returns:
            Liste de tokens.
        """
        output: List[str] = []
        for token in self.split_text(text):
            token = self.normalize_token(token)
            output.extend(self.tokenize_word(token, strip_eow=strip_eow))
        return output

    #* ------------------------------------------------------------------
    #* Sérialisation
    #* ------------------------------------------------------------------

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        """Sauvegarde le modèle BPE dans un fichier JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "end_of_word": self.end_of_word,
            "vocab": sorted(self.vocab),
            "merges": [[a, b] for a, b in self.merges],
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "Tokenizer":
        """Charge un modèle BPE depuis un fichier JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(end_of_word=data.get("end_of_word", "</w>"))
        tok.vocab = set(data.get("vocab", []))
        tok.vocab.update(tok.special_tokens)
        tok.merges = [tuple(x) for x in data.get("merges", [])]
        tok.merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        loaded = data.get("token_to_id")

        if loaded:
            tok.token_to_id = {str(t): int(i) for t, i in loaded.items()}
        else:
            tok.token_to_id = {t: i for i, t in enumerate(sorted(tok.vocab))}

        tok.id_to_token = {i: t for t, i in tok.token_to_id.items()}
        return tok

    #* ------------------------------------------------------------------
    #* Encoding
    #* ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_text(text)
        unk_id = self.token_to_id["<UNK>"]
        return [self.token_to_id.get(t, unk_id) for t in tokens]

    def encode_corpus(self, corpus_path: str, limit: int = 0) -> List[int]:
        """
        Encode un corpus JSONL complet en liste d'ids.

        Optimisations :
        ------------------------------------------------------------------
        | Cache word -> [ids]  | _apply_merges appelée une seule fois     |
        |                      | par forme de mot unique (hit ~100% après |
        |                      | quelques centaines de lignes)            |
        | array('i')           | Stockage contigu ; extend par batch      |
        |                      | ~3x plus rapide qu'une list + append     |
        ------------------------------------------------------------------

        Args:
            corpus_path: Chemin vers le fichier JSONL.
            limit: limite de lignes JSONL a encoder (0 = illimite).

        Returns:
            Liste d'entiers (token ids).
        """
        token_to_id = self.token_to_id
        unk_id = token_to_id["<UNK>"]
        split_text = self.split_text
        normalize_token = self.normalize_token
        tokenize_word = self.tokenize_word
        json_loads = json.loads

        # Cache raw_token -> [ids]
        word_cache: Dict[str, List[int]] = {}

        def token_ids(raw: str) -> List[int]:
            ids = word_cache.get(raw)
            if ids is None:
                ids = [
                    token_to_id.get(t, unk_id)
                    for t in tokenize_word(normalize_token(raw))
                ]
                word_cache[raw] = ids
            return ids

        buf = array("i")

        with open(corpus_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f, desc="Encoding corpus"), start=1):
                if limit and i > limit:
                    break
                line = line.strip()
                if not line:
                    continue
                text = json_loads(line).get("text", "")
                if not text:
                    continue
                for raw in split_text(text):
                    buf.extend(token_ids(raw))

        return buf.tolist()

    #* ------------------------------------------------------------------
    #* Accesseurs
    #* ------------------------------------------------------------------

    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> set[str]:
        return self.vocab

    def get_token_to_id(self) -> Dict[str, int]:
        return self.token_to_id

    def get_id_to_token(self) -> Dict[int, str]:
        return self.id_to_token