"""Implémentation optimisée de Byte Pair Encoding (BPE).

Optimisations clés vs version naïve :
─────────────────────────────────────────────────────────────────────────────
| Problème original          | Solution appliquée                           |
|----------------------------|----------------------------------------------|
| count_pairs() O(N) × V     | Comptage incrémental — seules les paires     |
|                            | voisines du token fusionné sont mises à jour |
| merge_pair() O(N) × V      | word_freq dict : on ne stocke chaque forme   |
|                            | de mot qu'une seule fois (× sa fréquence)    |
| most_frequent_pair() O(P)  | Heap max avec lazy-deletion -> O(log P)      |
─────────────────────────────────────────────────────────────────────────────

Complexité :
    Naïve   : O(V × N)          N = nb total tokens dans corpus
    Optimisée: O(V × W × log P) W = nb mots uniques (W << N en général)

Usage :
    python tokenizer.py
    from tokenizer.tokenizer import Tokenizer
    tok = Tokenizer.load("tokenizer/token/bpe_model.json")
    tokens = tok.tokenize_text("votre texte à tokeniser ici")
"""

from __future__ import annotations

import heapq
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

DEFAULT_MODEL_PATH = "tokenizer/token/bpe_model.json"
SPACE_TOKEN = "▁"


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

    # ------------------------------------------------------------------
    # Découpage et normalisation
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> List[str]:
        """Découpe texte en mots + ponctuation en conservant les espaces."""
        return re.findall(r"\s+|[\w]+|[^\w\s]", text, re.UNICODE)

    def normalize_token(self, token: str) -> str:
        return SPACE_TOKEN if token.isspace() else token

    # ------------------------------------------------------------------
    # Préparation corpus
    # ------------------------------------------------------------------

    def prepare_word(self, word: str) -> List[str]:
        """Convertit un mot en symboles caractères + marqueur fin de mot."""
        if not word:
            return [self.end_of_word]
        return list(word) + [self.end_of_word]

    def _build_word_freq(self, texts: Iterable[str]) -> Dict[Tuple[str, ...], int]:
        """
        Construit un dictionnaire { séquence_symboles -> fréquence }.

        Au lieu de stocker chaque occurrence du mot "the" 10 000 fois,
        on stocke  ('t','h','e','</w>') -> 10000.
        Cela réduit drastiquement le coût des itérations suivantes.
        """
        word_freq: Dict[Tuple[str, ...], int] = defaultdict(int)
        for text in tqdm(texts, desc="Preparing corpus"):
            for token in self.split_text(text):
                token = self.normalize_token(token)
                key = tuple(self.prepare_word(token))
                word_freq[key] += 1
        return word_freq

    # ------------------------------------------------------------------
    # Comptage de paires (helpers internes)
    # ------------------------------------------------------------------

    @staticmethod
    def _pairs_in_word(word: Tuple[str, ...]) -> List[Tuple[str, str]]:
        """_summary_
        Trouve toutes les paires de symboles adjacents dans un mot donné.

        Args:
            word (Tuple[str, ...]): Un mot représenté comme une séquence de symboles (caractères + </w>).

        Returns:
            List[Tuple[str, str]]: Liste de toutes les paires de symboles adjacents dans le mot.
        """
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    def _count_all_pairs(self, word_freq: Dict[Tuple[str, ...], int],) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], set]]:
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

    # ------------------------------------------------------------------
    # Training (version optimisée)
    # ------------------------------------------------------------------

    def train(self, texts: Iterable[str], target_vocab_size: int = 5000, min_pair_freq: int = 2, verbose: bool = False,) -> None:
        """
        Apprend les fusions BPE.

        Optimisations actives :
           word_freq dict   -> évite de re-parcourir les doublons de mots
           mise à jour incrémentale des pair_counts après chaque merge
           heap max (lazy-deletion) -> meilleure paire en O(log P)
        """
        # ── 1. Construction du dictionnaire de fréquences ────────────────
        word_freq = self._build_word_freq(texts)

        # ── 2. Vocabulaire initial (caractères uniques) ───────────────────
        self.vocab = set(self.special_tokens)
        for word in word_freq:
            self.vocab.update(word)
        self.merges = []
        self.merge_ranks = {}

        # ── 3. Comptage initial de toutes les paires ──────────────────────
        pair_counts, pair_to_words = self._count_all_pairs(word_freq)

        # ── 4. Heap max (on stocke des tuples (-fréquence, paire)) ────────
        #   Lazy-deletion : on pousse de nouvelles entrées sans retirer les
        #   anciennes ; on ignore celles dont le count stocké ne correspond
        #   plus à pair_counts[paire].
        heap: List[Tuple[int, Tuple[str, str]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        pbar = tqdm(total=target_vocab_size, desc="Training BPE")

        while len(self.vocab) < target_vocab_size:

            # ── Trouver la meilleure paire via le heap ─────────────────
            best_pair: Optional[Tuple[str, str]] = None
            best_freq = 0

            while heap:
                neg_cnt, candidate = heapq.heappop(heap)
                # Lazy-deletion : on vérifie que l'entrée est encore valide
                if pair_counts.get(candidate, 0) == -neg_cnt:
                    best_pair = candidate
                    best_freq = -neg_cnt
                    break

            if best_pair is None or best_freq < min_pair_freq:
                if verbose:
                    reason = "plus de paires" if best_pair is None else f"fréquence trop faible ({best_freq})"
                    print(f"\n[BPE] Arrêt : {reason}")
                break

            # ── Fusionner ─────────────────────────────────────────────
            a, b = best_pair
            merged = a + b
            self.vocab.add(merged)
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1
            pbar.update(1)

            # ── Mise à jour incrémentale ──────────────────────────────
            #   Pour chaque mot contenant (a, b) :
            #     1. Soustraire les fréquences des anciennes paires
            #     2. Construire le nouveau mot fusionné
            #     3. Ajouter les fréquences des nouvelles paires
            #   -> seule une fraction du corpus est re-visitée
            affected_words = list(pair_to_words.pop(best_pair, set()))

            for old_word in affected_words:
                freq = word_freq.get(old_word)
                if freq is None:
                    continue  # déjà supprimé lors d'une itération précédente

                # — Décrémenter les paires de l'ancien mot ——————————————
                for pair in self._pairs_in_word(old_word):
                    cnt = pair_counts.get(pair, 0) - freq
                    if cnt <= 0:
                        pair_counts.pop(pair, None)
                        pair_to_words.pop(pair, None)
                    else:
                        pair_counts[pair] = cnt
                        pair_to_words[pair].discard(old_word)

                # — Construire le nouveau mot ——————————————————————————
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

                # — Fusionner avec une éventuelle entrée existante ———————
                del word_freq[old_word]
                word_freq[new_word] = word_freq.get(new_word, 0) + freq

                # — Incrémenter les paires du nouveau mot ————————————————
                for pair in self._pairs_in_word(new_word):
                    new_cnt = pair_counts.get(pair, 0) + freq
                    pair_counts[pair] = new_cnt
                    pair_to_words[pair].add(new_word)
                    # On pousse la nouvelle valeur dans le heap (lazy)
                    heapq.heappush(heap, (-new_cnt, pair))

            if verbose and len(self.merges) % 100 == 0:
                print(
                    f"[BPE] merges={len(self.merges)} | vocab={len(self.vocab)} "
                    f"| best={best_pair} ({best_freq})"
                )

        pbar.close()

        # ── Finalisation ──────────────────────────────────────────────────
        self.token_to_id = {tok: i for i, tok in enumerate(sorted(self.vocab))}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    # ------------------------------------------------------------------
    # Tokenisation (inchangée)
    # ------------------------------------------------------------------

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """_summary_
        Applique les merges appris à une séquence de symboles (caractères + </w>).
        Args:
            symbols (List[str]): Liste de symboles représentant un mot (ex: ['t', 'h', 'e', '</w>']).

        Returns:
            List[str]: Liste de tokens après application des merges (ex: ['the</w>'] si 't'+'h'+'e' a été fusionné).
        """
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [
                (self.merge_ranks[p], i)
                for i, p in enumerate(pairs)
                if p in self.merge_ranks
            ]
            if not ranked:
                break
            _, idx = min(ranked)
            symbols = symbols[:idx] + [symbols[idx] + symbols[idx + 1]] + symbols[idx + 2:]
        return symbols

    def tokenize_word(self, word: str, strip_eow: bool = True) -> List[str]:
        """_summary_
        Tokenise un mot donné en appliquant les merges appris.

        Args:
            word (str): Le mot à tokeniser.
            strip_eow (bool, optional): Indique si le marqueur de fin de mot doit être retiré des tokens finaux. Par défaut à True.

        Returns:
            List[str]: Liste de tokens résultant de la tokenisation du mot.
        """
        symbols = self.prepare_word(word)
        tokens = self._apply_merges(symbols)
        if strip_eow:
            tokens = [t.replace(self.end_of_word, "") for t in tokens if t != self.end_of_word]
        return tokens

    def tokenize_text(self, text: str, strip_eow: bool = True) -> List[str]:
        """_summary_
        Tokenise un texte complet en appliquant les merges appris à chaque mot.

        Args:
            text (str): Le texte à tokeniser.
            strip_eow (bool, optional): Indique si le marqueur de fin de mot doit être retiré des tokens finaux. Par défaut à True.

        Returns:
            List[str]: Liste de tokens résultant de la tokenisation du texte.
        """
        output: List[str] = []
        for token in self.split_text(text):
            token = self.normalize_token(token)
            output.extend(self.tokenize_word(token, strip_eow=strip_eow))
        return output

    # ------------------------------------------------------------------
    # Sérialisation (inchangée)
    # ------------------------------------------------------------------

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        """_summary_
        Sauvegarde le modèle BPE (vocabulaire + merges) dans un fichier JSON.

        Args:
            path (str, optional): Chemin du fichier de sauvegarde. Par défaut à "tokenizer/token/bpe_model.json".
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"end_of_word": self.end_of_word, "vocab": sorted(self.vocab), "merges": [[a, b] for a, b in self.merges],"token_to_id": self.token_to_id}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "Tokenizer":
        """_summary_
        Charge un modèle BPE à partir d'un fichier JSON.
        
        Args:
            path (str, optional): Chemin du fichier de chargement. Par défaut à "tokenizer/token/bpe_model.json".

        Returns:
            Tokenizer: Instance de Tokenizer initialisée avec le vocabulaire et les merges chargés.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # mise en place des données dans la classe (voir __init__)
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

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize_text(text)
        unk_id = self.token_to_id["<UNK>"]
        return [self.token_to_id.get(t, unk_id) for t in tokens]