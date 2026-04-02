# -*- coding: utf-8 -*-
"""
Construit un dataset JSONL propre depuis la base crawler SQLite.

Ce script:
- lit toutes les pages depuis crawler.db,
- nettoie agressivement le texte,
- filtre les entrées trop bruitées,
- découpe en chunks de taille raisonnable,
- déduplique les chunks,
- supprime puis recrée entièrement dataset.jsonl.

Usage:
    python build_clean_dataset.py
    python build_clean_dataset.py --db data/datasets/crawler.db --jsonl data/datasets/dataset.jsonl
    python build_clean_dataset.py --chunk-size 700 --chunk-overlap 80 --min-chars 180
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
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))

from Crawler.config import CRAWLER_CONFIG


RE_MULTI_SPACE = re.compile(r"[ \t\f\v]+")
RE_MULTI_NEWLINES = re.compile(r"\n{3,}")
RE_EMPTY_PARENS = re.compile(r"\(\s*\)")
RE_MARKDOWN_TABLE = re.compile(r"^\s*\|.*\|\s*$")
RE_MARKDOWN_SEP = re.compile(r"^\s*\|?\s*[:-]{2,}.*$")
RE_LIST_BULLET = re.compile(r"^\s*[-*•]+\s+")
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_WORD = re.compile(r"\b\w+\b", re.UNICODE)
RE_SENTENCE_END = re.compile(r"[.!?]\s+")
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_LONG_PUNCT = re.compile(r"([!?.,:;()\[\]{}\-|_/*+=])\1{3,}")

MAX_ANALYSIS_CHARS = 120_000
MAX_CHARS_PER_PAGE = 2_000_000

# Motifs très fréquents dans le bruit web/SEO/CTA.
NOISE_PATTERNS = [
    re.compile(r"\b(thunderbit|click here|cliquez ici|inscrivez-vous|subscribe|abonnez-vous)\b", re.IGNORECASE),
    re.compile(r"\b(pr[êe]t\s*\?\s*c'?est\s*parti)\b", re.IGNORECASE),
    re.compile(r"\b(essayez gratuitement|free trial|d[ée]mo)\b", re.IGNORECASE),
    re.compile(r"\b(cookie|politique de confidentialit[ée]|privacy policy|terms of service)\b", re.IGNORECASE),
    re.compile(r"\b(share on|partager sur|follow us|suivez-nous)\b", re.IGNORECASE),
    re.compile(r"\b(acheter|buy now|ajouter au panier|add to cart)\b", re.IGNORECASE),
    re.compile(r"\b(questions fréquentes|faq|contactez-nous|contact us)\b", re.IGNORECASE),
    re.compile(r"\b(©|copyright|all rights reserved|tous droits réservés)\b", re.IGNORECASE),
    re.compile(r"\b(logo|icon|image|photo|screenshot)\b", re.IGNORECASE),
    re.compile(r"\b(ce site|this site|notre site|our site)\b", re.IGNORECASE),
    re.compile(r"\b(202[0-4]|20[0-1][0-9]|19[5-9][0-9])\b"),  # années récentes, souvent dans les pieds de page
    re.compile(r"\b(page \d+ of \d+|page \d+ sur \d+)\b", re.IGNORECASE),
    re.compile(r"\b(terms and conditions|terms & conditions|conditions générales)\b", re.IGNORECASE),
    re.compile(r"\b(privacy policy|politique de confidentialité|cookie policy|politique des cookies)\b", re.IGNORECASE),
    re.compile(r"\b(about us|à propos|notre équipe|our team)\b", re.IGNORECASE),
    re.compile(r"\b(services|produits|solutions|contact|support|aide|help)\b", re.IGNORECASE),
    re.compile(r"\b(Editions|Contact|Mentions légales|Legal notice|Sitemap|Plan du site)\b", re.IGNORECASE),
    re.compile(r"\b(©|®|™|all rights reserved|tous droits réservés)\b", re.IGNORECASE),
    re.compile(r"\b(ce site|this site|notre site|our site)\b", re.IGNORECASE),
    re.compile(r"\b(202[0-4]|20[0-1][0-9]|19[5-9][0-9])\b"),  # années récentes, souvent dans les pieds de page
]


@dataclass
class BuildStats:
    pages_total: int = 0
    pages_with_text: int = 0
    pages_kept_after_clean: int = 0
    chunks_written: int = 0
    dropped_empty: int = 0
    dropped_low_quality: int = 0
    dropped_duplicates: int = 0
    dropped_noise_lines: int = 0


def normalize_text(text: str) -> tuple[str, int]:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = RE_CONTROL.sub(" ", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = RE_EMPTY_PARENS.sub(" ", text)
    text = RE_LONG_PUNCT.sub(r"\1\1", text)

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

        lowered = line.lower()
        if any(pattern.search(lowered) for pattern in NOISE_PATTERNS):
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


def quality_score(text: str, min_chars: int) -> bool:
    if not text or len(text) < min_chars:
        return False

    # Analyse sur une fenêtre bornée pour éviter les ralentissements sur textes géants.
    sample = text[:MAX_ANALYSIS_CHARS]

    word_count = 0
    for _ in RE_WORD.finditer(sample):
        word_count += 1
        if word_count >= 30:
            break

    if word_count < 30:
        return False

    letters = sum(ch.isalpha() for ch in sample)
    if letters / max(len(sample), 1) < 0.55:
        return False

    punct = sum(ch in "!?.,:;()[]{}|/\\-_" for ch in sample)
    if punct / max(len(sample), 1) > 0.20:
        return False

    if text.count("(") + text.count(")") > max(12, len(text) // 120):
        return False

    if "()" in text:
        return False

    return True


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = RE_SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if len(chunk_words) < max(80, chunk_size // 4):
            continue
        chunks.append(" ".join(chunk_words))
    return chunks


def iter_rows(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT id, url, title, content, domain, word_count FROM pages ORDER BY id"
        for row in conn.execute(query):
            yield row
    finally:
        conn.close()


def rebuild_dataset(
    db_path: str,
    output_jsonl: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
    max_pages: int,
    progress_every: int,
) -> BuildStats:
    stats = BuildStats()
    seen_hashes: set[str] = set()

    out_dir = os.path.dirname(output_jsonl) or "."
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_jsonl):
        try:
            os.remove(output_jsonl)
        except PermissionError:
            # Sous Windows, un process peut garder le fichier ouvert en lecture.
            # On continue: le mode "w" ci-dessous tronquera le contenu si possible.
            pass

    try:
        f_out = open(output_jsonl, "w", encoding="utf-8")
    except PermissionError as e:
        raise PermissionError(
            f"Impossible de recréer {output_jsonl}: fichier verrouillé par un autre process. "
            "Ferme les scripts/processus qui lisent ce JSONL puis relance."
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
                # On ignore les pages anormalement massives qui bloquent le preprocessing.
                stats.dropped_low_quality += 1
                continue

            stats.pages_with_text += 1
            cleaned_text, dropped_lines = normalize_text(raw_text)
            stats.dropped_noise_lines += dropped_lines

            if not quality_score(cleaned_text, min_chars=min_chars):
                stats.dropped_low_quality += 1
                continue

            stats.pages_kept_after_clean += 1

            # Découpage pour limiter les séquences trop longues et stabiliser l'entraînement.
            chunks = chunk_text(
                cleaned_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk in chunks:
                if not quality_score(chunk, min_chars=max(120, min_chars // 2)):
                    stats.dropped_low_quality += 1
                    continue

                chunk_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                if chunk_hash in seen_hashes:
                    stats.dropped_duplicates += 1
                    continue
                seen_hashes.add(chunk_hash)

                record = {
                    "text": chunk,
                    "url": row["url"] or "",
                    "title": row["title"] or "",
                    "domain": row["domain"] or "",
                    "source_id": int(row["id"]),
                    "word_count": len(chunk.split()),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.chunks_written += 1

            if progress_every > 0 and (stats.pages_total % progress_every == 0):
                print(
                    f"[progress] pages={stats.pages_total} "
                    f"kept={stats.pages_kept_after_clean} chunks={stats.chunks_written}"
                )

    return stats


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
        help="Taille max d'un chunk en mots (approx. 512-1024 tokens)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help="Chevauchement en mots entre chunks",
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
        help="Afficher un log de progression toutes les N pages",
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

    print("=" * 72)
    print("  REBUILD DATASET CLEAN")
    print("=" * 72)
    print(f"DB source            : {args.db}")
    print(f"JSONL sortie         : {args.jsonl}")
    print(f"Chunk size (mots)    : {args.chunk_size}")
    print(f"Chunk overlap (mots) : {args.chunk_overlap}")
    print(f"Min chars            : {args.min_chars}")
    print(f"Max pages            : {args.max_pages if args.max_pages > 0 else 'all'}")
    print(f"Progress every       : {args.progress_every}")

    stats = rebuild_dataset(
        db_path=args.db,
        output_jsonl=args.jsonl,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
        max_pages=args.max_pages,
        progress_every=args.progress_every,
    )

    print("\nRésultats")
    print("-" * 72)
    print(f"Pages totales                : {stats.pages_total}")
    print(f"Pages avec texte             : {stats.pages_with_text}")
    print(f"Pages gardées après nettoyage: {stats.pages_kept_after_clean}")
    print(f"Chunks écrits                : {stats.chunks_written}")
    print(f"Rejets vides                 : {stats.dropped_empty}")
    print(f"Rejets faible qualité        : {stats.dropped_low_quality}")
    print(f"Rejets doublons              : {stats.dropped_duplicates}")
    print(f"Lignes bruit supprimées      : {stats.dropped_noise_lines}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
