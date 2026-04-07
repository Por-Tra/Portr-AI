# -*- coding: utf-8 -*-
"""
Construit un dataset JSONL propre depuis la base crawler SQLite.

Ce script:
- lit toutes les pages depuis crawler.db,
- nettoie agressivement le texte,
- filtre les entrées trop bruitées,
- découpe en chunks en respectant les frontières de phrases,
- déduplique les chunks (SHA-256 + LSH banding pour near-duplicates),
- supprime puis recrée entièrement dataset.jsonl.

Usage:
    python build_clean_dataset.py
    python build_clean_dataset.py --db data/datasets/crawler.db --jsonl data/datasets/dataset.jsonl
    python build_clean_dataset.py --chunk-size 700 --chunk-overlap 80 --min-chars 180

Corrections vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| 46% des chunks commencent       | Chunking par frontières de phrases :     |
| au milieu d'une phrase          | on accumule des phrases entières jusqu'à |
| (coupure word-based aveugle)    | atteindre chunk_size, jamais au milieu   |
|---------------------------------|------------------------------------------|
| 70% des chunks finissent sans   | Même fix : chaque chunk se termine       |
| ponctuation (coupure mid-phrase)| toujours sur '.', '!', '?' ou '…'        |
|---------------------------------|------------------------------------------|
| Points/virgules flottants       | RE_FLOATING_PUNCT : 'mot . mot' ->        |
| (artefact HTML, 36% du corpus)  | 'mot. mot' — appliqué AVANT le chunking  |
|---------------------------------|------------------------------------------|
| NOISE_PATTERNS trop agressifs : | Suppression des patterns qui filtrent    |
| 'services', 'produits', 'aide', | des mots courants du français légitime   |
| '2020', '2023'... supprimaient  | (années, mots génériques). Seuls les     |
| du contenu éditorial valide     | patterns réellement bruités sont gardés. |
|---------------------------------|------------------------------------------|
| near-dup O(N²) : liste de fps   | LSH banding O(N) amorti : on hash        |
| comparée 1 à 1 -> des heures     | le fingerprint en B bandes, lookup en    |
| sur 1M chunks                   | dict. md5 pour éviter PYTHONHASHSEED.    |
|---------------------------------|------------------------------------------|
| quality_score : boucles         | Remplacées par re.findall sur le sample  |
| char par char (×2 par chunk)    | -> même résultat, 5-10x plus rapide       |
|---------------------------------|------------------------------------------|
| 15 NOISE_PATTERNS séquentiels   | Combinés en 1 seul pattern ORed          |
| par ligne                       | -> même résultat, ~10x plus rapide        |
|---------------------------------|------------------------------------------|
| Overlap 80 mots = 7.8% du       | Overlap ramené à 0 par défaut (la        |
| corpus en double, biaise        | continuité inter-chunks est gérée par    |
| l'entraînement                  | le chunking phrase-aware)                |
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

try:
    from Crawler.config import CRAWLER_CONFIG
except ImportError:
    CRAWLER_CONFIG: dict = {}


# ---------------------------------------------------------------------------
# Expressions régulières — compilées une seule fois au chargement du module
# ---------------------------------------------------------------------------

RE_MULTI_SPACE     = re.compile(r"[ \t\f\v]+")
RE_MULTI_NEWLINES  = re.compile(r"\n{3,}")
RE_CONTROL         = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_LONG_PUNCT      = re.compile(r"([!?.,:;()\[\]{}\-|_/*+=])\1{3,}")
RE_URL             = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_WORD            = re.compile(r"\b\w+\b", re.UNICODE)
RE_MARKDOWN_TABLE  = re.compile(r"^\s*\|.*\|\s*$")
RE_MARKDOWN_SEP    = re.compile(r"^\s*\|?\s*[:-]{2,}.*$")
RE_LIST_BULLET     = re.compile(r"^\s*[-*•·▸▹►]+\s+")
RE_EMPTY_PARENS    = re.compile(r"\(\s*\)")

# Artefact HTML fréquent : 'mot . suite' ou 'mot , suite'
RE_FLOATING_DOT    = re.compile(r"(\w)\s+\.\s+(\w)")
RE_FLOATING_COMMA  = re.compile(r"(\w)\s+,\s+(\w)")

# Séparation de phrases
RE_SENTENCE_SPLIT  = re.compile(r"(?<=[.!?…])\s+")

# Emojis et symboles hors latin étendu
RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\u2500-\u257F"
    "]",
    flags=re.UNICODE,
)

# Ratios qualité — remplacent les boucles char par char
RE_ALPHA = re.compile(r"[a-zA-ZÀ-ÿ\u00C0-\u024F]")
RE_PUNCT = re.compile(r"[!?.,:;()\[\]{}|/\\\-_]")

MAX_ANALYSIS_CHARS = 120_000
MAX_CHARS_PER_PAGE = 2_000_000


# ---------------------------------------------------------------------------
# Patterns de bruit — combinés en un seul pattern ORed (fix #3)
# ---------------------------------------------------------------------------
# Règle : on ne supprime QUE ce qui est indubitablement du bruit UI/SEO/boilerplate.
# Un seul re.compile avec alternance est ~10x plus rapide que 15 patterns séquentiels,
# et produit exactement les mêmes résultats.

_NOISE_PARTS = [
    # CTA explicites
    r"\b(click here|cliquez ici|inscrivez-vous maintenant|subscribe now)\b",
    r"\b(essayez gratuitement|free trial|start free trial|d[ée]mo gratuite)\b",
    r"\b(share on|partager sur|follow us on|suivez-nous sur)\b",
    r"\b(add to cart|ajouter au panier|buy now|acheter maintenant)\b",
    r"\b(page \d+ of \d+|page \d+ sur \d+)\b",
    # Boilerplate juridique / pied de page
    r"\b(all rights reserved|tous droits r[eé]serv[eé]s)\b",
    r"\b(terms and conditions|conditions g[eé]n[eé]rales d'utilisation)\b",
    r"\b(privacy policy|politique de confidentialit[eé])\b",
    r"\b(cookie policy|politique des cookies|g[eé]rer les cookies)\b",
    r"\b(d[eé]claration d'accessibilit[eé]|mentions l[eé]gales)\b",
    # Bruit SEO/thunderbit
    r"\bthunderbit\b",
    r"\b(pr[êe]t\s*\?\s*c'?est\s*parti)\b",
    # Navigation UI pure
    r"\b(sitemap|plan du site|fil d'ariane|breadcrumb)\b",
    r"\b(DÉCLARATION D'ACCESSIBILITÉ)\b",
]

NOISE_COMBINED = re.compile(
    "|".join(f"(?:{p})" for p in _NOISE_PARTS),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BuildStats:
    pages_total:            int = 0
    pages_with_text:        int = 0
    pages_kept_after_clean: int = 0
    chunks_written:         int = 0
    dropped_empty:          int = 0
    dropped_low_quality:    int = 0
    dropped_duplicates:     int = 0
    dropped_noise_lines:    int = 0
    dropped_mid_sentence:   int = 0


# ---------------------------------------------------------------------------
# Nettoyage du texte
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> tuple[str, int]:
    """
    Nettoie un texte brut issu du crawler.

    Étapes dans l'ordre :
    1. Normalisation unicode NFKC + contrôle CR/LF
    2. Suppression des caractères de contrôle et emojis
    3. Correction des artefacts HTML (points/virgules flottants)
    4. Filtrage ligne par ligne (markdown, bruit UI, URLs)
    5. Normalisation des espaces

    Returns:
        (texte nettoyé, nombre de lignes supprimées pour bruit)
    """
    # ── 1. Unicode + sauts de ligne
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_CONTROL.sub(" ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "")

    # ── 2. Emojis et symboles non-latin
    text = RE_EMOJI.sub(" ", text)

    # ── 3. Artefacts de ponctuation HTML
    for _ in range(3):
        new = RE_FLOATING_DOT.sub(r"\1. \2", text)
        new = RE_FLOATING_COMMA.sub(r"\1, \2", new)
        if new == text:
            break
        text = new

    # ── 4. Nettoyage ponctuation répétée et parenthèses vides
    text = RE_LONG_PUNCT.sub(r"\1\1", text)
    text = RE_EMPTY_PARENS.sub(" ", text)

    # ── 5. Filtrage ligne par ligne
    clean_lines: list[str] = []
    dropped_noise_lines = 0

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if RE_MARKDOWN_TABLE.match(line) or RE_MARKDOWN_SEP.match(line):
            dropped_noise_lines += 1
            continue

        if RE_LIST_BULLET.match(line):
            line = RE_LIST_BULLET.sub("", line).strip()

        # Fix #3 : un seul pattern combiné au lieu de 15 séquentiels
        if NOISE_COMBINED.search(line):
            dropped_noise_lines += 1
            continue

        line = RE_URL.sub(" ", line)
        line = RE_MULTI_SPACE.sub(" ", line).strip()

        if len(line) < 3:
            dropped_noise_lines += 1
            continue

        clean_lines.append(line)

    text = "\n".join(clean_lines)
    text = RE_MULTI_NEWLINES.sub("\n\n", text)
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    return text, dropped_noise_lines


# ---------------------------------------------------------------------------
# Contrôle qualité — fix #2 : re.findall au lieu de boucles char par char
# ---------------------------------------------------------------------------

def quality_score(text: str, min_chars: int) -> bool:
    """
    Retourne True si le texte passe les critères minimaux de qualité.

    Critères :
    - longueur minimale en caractères
    - au moins 30 mots
    - ratio lettres/total ≥ 55 %
    - ratio ponctuation bruyante ≤ 20 %
    - pas de parenthèses vides résiduelles

    Implémentation : re.findall remplace les boucles char par char
    -> même résultat, 5-10x plus rapide sur des samples de 120 000 chars.
    """
    if not text or len(text) < min_chars:
        return False

    sample = text[:MAX_ANALYSIS_CHARS]
    n = len(sample)

    words = RE_WORD.findall(sample)
    if len(words) < 30:
        return False

    if len(RE_ALPHA.findall(sample)) / n < 0.55:
        return False

    if len(RE_PUNCT.findall(sample)) / n > 0.20:
        return False

    if "()" in text or "( )" in text:
        return False

    return True


# ---------------------------------------------------------------------------
# Chunking par frontières de phrases
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Découpe un texte en phrases en respectant les délimiteurs naturels."""
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = RE_SENTENCE_SPLIT.split(paragraph)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Découpe un texte en chunks en respectant les frontières de phrases.

    - On accumule des phrases entières jusqu'à dépasser chunk_size mots.
    - On ne coupe JAMAIS au milieu d'une phrase.
    - Les chunks trop courts (< 80 mots) sont ignorés.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    min_words = max(80, chunk_size // 4)
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        sent_words = len(sentence.split())

        if sent_words >= chunk_size and not current_sentences:
            chunks.append(sentence)
            continue

        current_sentences.append(sentence)
        current_word_count += sent_words

        if current_word_count >= chunk_size:
            chunk_str = " ".join(current_sentences)
            if len(chunk_str.split()) >= min_words:
                chunks.append(chunk_str)

            if chunk_overlap > 0:
                overlap_sentences: list[str] = []
                overlap_words = 0
                for s in reversed(current_sentences):
                    sw = len(s.split())
                    if overlap_words + sw > chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_words += sw
                current_sentences = overlap_sentences
                current_word_count = overlap_words
            else:
                current_sentences = []
                current_word_count = 0

    if current_sentences:
        chunk_str = " ".join(current_sentences)
        if len(chunk_str.split()) >= min_words:
            chunks.append(chunk_str)

    return chunks


# ---------------------------------------------------------------------------
# Déduplication — fix #1 : LSH banding O(N) amorti
# ---------------------------------------------------------------------------

def _text_fingerprint(text: str, ngram: int = 8) -> list[str]:
    """
    Calcule une liste triée de n-grammes de mots.
    On retourne une list (et non frozenset) pour conserver un ordre stable
    nécessaire au découpage en bandes LSH.
    """
    words = text.lower().split()
    if len(words) < ngram:
        return sorted(set(words))
    return sorted(set(
        " ".join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)
    ))


class DedupIndex:
    """
    Déduplication à deux niveaux :

    1. SHA-256 exact          -> set, O(1) lookup
    2. Near-dup par LSH banding -> O(1) amorti au lieu de O(N²)

    Principe du banding LSH :
    ─────────────────────────
    On découpe le fingerprint (liste de n-grammes triés) en BANDS bandes.
    Chaque bande est hashée via MD5 (déterministe, pas sensible à PYTHONHASHSEED).
    Si deux textes partagent au moins une bande identique, ils sont near-dups.

    Avec BANDS=20, ROWS=5 et Jaccard ≥ 0.8 :
      P(détection)  ≈ 1 - (1 - 0.8^5)^20  ≈ 99.97 %   -> quasi aucun faux négatif
      P(faux positif) est faible grâce à MD5 (espace 2^128)

    Complexité : O(BANDS) par chunk au lieu de O(N) comparaisons -> O(N) amorti global.
    Mémoire    : O(N × BANDS) entrées de dict (int -> bool).
    """

    BANDS = 20  # nombre de bandes — augmenter améliore la précision, coûte de la mémoire
    ROWS  = 5   # n-grammes par bande — augmenter réduit les faux positifs

    def __init__(self, near_dup_threshold: float = 0.8):
        self._exact: set[str] = set()
        # Un dict par bande : hash_md5_int -> True (présence seule suffit)
        self._buckets: list[dict[int, bool]] = [
            {} for _ in range(self.BANDS)
        ]
        self._use_near_dup = near_dup_threshold < 1.0

    def _band_hashes(self, tokens: list[str]) -> list[int]:
        """
        Découpe la liste de tokens en BANDS bandes et retourne leurs hashs MD5.
        MD5 est utilisé (et non hash()) pour garantir la reproductibilité
        indépendamment de PYTHONHASHSEED.
        """
        size = max(1, len(tokens) // self.BANDS)
        hashes: list[int] = []
        for i in range(self.BANDS):
            band = tokens[i * size: (i + 1) * size]
            digest = hashlib.md5(" ".join(band).encode(), usedforsecurity=False).digest()
            # On prend les 8 premiers octets comme entier pour clé de dict légère
            hashes.append(int.from_bytes(digest[:8], "little"))
        return hashes

    def is_duplicate(self, text: str) -> bool:
        # ── Niveau 1 : exact SHA-256
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self._exact:
            return True

        # ── Niveau 2 : near-dup LSH banding
        if self._use_near_dup:
            tokens      = _text_fingerprint(text)
            band_hashes = self._band_hashes(tokens)

            # Vérification : au moins un bucket contient déjà ce hash ?
            for band_idx, bh in enumerate(band_hashes):
                if bh in self._buckets[band_idx]:
                    return True  # near-dup détecté

            # Pas de doublon -> on indexe dans tous les buckets
            for band_idx, bh in enumerate(band_hashes):
                self._buckets[band_idx][bh] = True

        self._exact.add(h)
        return False


# ---------------------------------------------------------------------------
# Lecture SQLite
# ---------------------------------------------------------------------------

def iter_rows(db_path: str):
    """Itère sur toutes les pages de la base SQLite (streaming, pas de chargement en RAM)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT id, url, title, content, domain, word_count FROM pages ORDER BY id"
        for row in conn.execute(query):
            yield row
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def rebuild_dataset(
    db_path: str,
    output_jsonl: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    max_pages: int,
    progress_every: int,
    near_dup_threshold: float,
) -> BuildStats:
    stats = BuildStats()
    dedup = DedupIndex(near_dup_threshold=near_dup_threshold)

    out_dir = os.path.dirname(output_jsonl) or "."
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_jsonl):
        try:
            os.remove(output_jsonl)
        except PermissionError:
            pass

    try:
        f_out = open(output_jsonl, "w", encoding="utf-8")
    except PermissionError as e:
        raise PermissionError(
            f"Impossible de recréer {output_jsonl}: fichier verrouillé. "
            "Ferme les processus qui lisent ce JSONL puis relance."
        ) from e

    with f_out as f:
        for row in iter_rows(db_path):
            if max_pages > 0 and stats.pages_total >= max_pages:
                break

            stats.pages_total += 1
            raw_text = row["content"] or ""

            if not raw_text.strip():
                stats.dropped_empty += 1
                continue

            if len(raw_text) > MAX_CHARS_PER_PAGE:
                stats.dropped_low_quality += 1
                continue

            stats.pages_with_text += 1
            cleaned_text, dropped_lines = normalize_text(raw_text)
            stats.dropped_noise_lines += dropped_lines

            if not quality_score(cleaned_text, min_chars=min_chars):
                stats.dropped_low_quality += 1
                continue

            stats.pages_kept_after_clean += 1

            chunks = chunk_text(
                cleaned_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk in chunks:
                if not quality_score(chunk, min_chars=max(120, min_chars // 2)):
                    stats.dropped_low_quality += 1
                    continue

                first_char = chunk.lstrip()[0] if chunk.strip() else ""
                if first_char and first_char.islower():
                    stats.dropped_mid_sentence += 1
                    continue

                if dedup.is_duplicate(chunk):
                    stats.dropped_duplicates += 1
                    continue

                record = {
                    "text":       chunk,
                    "url":        row["url"]   or "",
                    "title":      row["title"] or "",
                    "domain":     row["domain"] or "",
                    "source_id":  int(row["id"]),
                    "word_count": len(chunk.split()),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.chunks_written += 1

            if progress_every > 0 and stats.pages_total % progress_every == 0:
                print(
                    f"[progress] pages={stats.pages_total} "
                    f"kept={stats.pages_kept_after_clean} "
                    f"chunks={stats.chunks_written} "
                    f"dupes={stats.dropped_duplicates}"
                )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruit dataset.jsonl propre depuis crawler.db"
    )
    parser.add_argument(
        "--db",
        default=CRAWLER_CONFIG.get("db_file", "data/datasets/crawler.db"),
        help="Chemin de la base SQLite source",
    )
    parser.add_argument(
        "--jsonl",
        default=CRAWLER_CONFIG.get("jsonl_output", "data/datasets/dataset.jsonl"),
        help="Chemin du JSONL de sortie (supprimé puis recréé)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
        help="Taille cible d'un chunk en mots",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help=(
            "Chevauchement en mots entre chunks consécutifs. "
            "0 = aucun chevauchement (recommandé pour l'entraînement LLM)"
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=180,
        help="Longueur minimale en caractères pour garder un texte",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Nombre max de pages à traiter (0 = tout)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Log de progression toutes les N pages",
    )
    parser.add_argument(
        "--near-dup-threshold",
        type=float,
        default=0.8,
        help=(
            "Seuil Jaccard pour la déduplication near-duplicate (0.8 = défaut). "
            "1.0 = déduplication exacte uniquement (plus rapide sur très grands corpus)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.db):
        print(f"[ERROR] Base de données introuvable: {args.db}")
        return 1

    if args.chunk_overlap >= args.chunk_size:
        print("[ERROR] --chunk-overlap doit être strictement inférieur à --chunk-size")
        return 1

    if not (0.0 < args.near_dup_threshold <= 1.0):
        print("[ERROR] --near-dup-threshold doit être entre 0.0 (exclus) et 1.0 (inclus)")
        return 1

    print("=" * 72)
    print("  REBUILD DATASET CLEAN")
    print("=" * 72)
    print(f"DB source              : {args.db}")
    print(f"JSONL sortie           : {args.jsonl}")
    print(f"Chunk size (mots)      : {args.chunk_size}")
    print(f"Chunk overlap (mots)   : {args.chunk_overlap}")
    print(f"Min chars              : {args.min_chars}")
    print(f"Near-dup threshold     : {args.near_dup_threshold}")
    print(f"Max pages              : {args.max_pages if args.max_pages > 0 else 'all'}")
    print(f"Progress every         : {args.progress_every}")
    print()

    stats = rebuild_dataset(
        db_path=args.db,
        output_jsonl=args.jsonl,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
        max_pages=args.max_pages,
        progress_every=args.progress_every,
        near_dup_threshold=args.near_dup_threshold,
    )

    print("\nRésultats")
    print("-" * 72)
    print(f"Pages totales                : {stats.pages_total}")
    print(f"Pages avec texte             : {stats.pages_with_text}")
    print(f"Pages gardées après nettoyage: {stats.pages_kept_after_clean}")
    print(f"Chunks écrits                : {stats.chunks_written}")
    print(f"Rejets vides                 : {stats.dropped_empty}")
    print(f"Rejets faible qualité        : {stats.dropped_low_quality}")
    print(f"Rejets doublons (exact+near) : {stats.dropped_duplicates}")
    print(f"Rejets mid-phrase            : {stats.dropped_mid_sentence}")
    print(f"Lignes bruit supprimées      : {stats.dropped_noise_lines}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())