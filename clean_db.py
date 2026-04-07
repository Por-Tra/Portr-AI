# -*- coding: utf-8 -*-
"""
Script de nettoyage et d'audit de la base de données crawler.

Actions disponibles :
    python clean_db.py --stats          Statistiques générales
    python clean_db.py --dupes          Lister les doublons de contenu
    python clean_db.py --remove-dupes   Supprimer les doublons (garde la 1ère occurrence)
    python clean_db.py --remove-short N Supprimer les pages avec < N mots (défaut 50)
    python clean_db.py --remove-nulls   Supprimer les pages sans contenu
    python clean_db.py --export-jsonl   Ré-exporter le dataset.jsonl depuis la DB
    python clean_db.py --vacuum         VACUUM SQLite (réduit la taille du fichier)
    python clean_db.py --all            Tout faire (audit + nettoyage complet)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from Crawler.config import CRAWLER_CONFIG
from data.DataBase.database import Database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char="=", n=65):
    print(char * n)


def cmd_stats(db: Database):
    _sep()
    print("  STATISTIQUES DE LA BASE")
    _sep()
    stats = db.get_statistics()
    if not stats:
        print("  Base vide ou erreur.")
        return
    print(f"  Pages totales          : {stats['total_pages']}")
    print(f"  Domaines uniques       : {stats['unique_domains']}")
    print(f"  Moy. mots / page       : {stats['avg_words']}")
    print(f"  Max mots / page        : {stats['max_words']}")
    print(f"  Liens totaux           : {stats['total_links']}")

    # Pages sans contenu
    nulls = db.conn.execute(
        "SELECT COUNT(*) FROM pages WHERE content IS NULL OR content = ''"
    ).fetchone()[0]
    print(f"  Pages sans contenu     : {nulls}")

    # Pages sans hash (crawlées avant la migration)
    no_hash = db.conn.execute(
        "SELECT COUNT(*) FROM pages WHERE content_hash IS NULL"
    ).fetchone()[0]
    print(f"  Pages sans hash        : {no_hash}")

    # Doublons de hash
    dup_count = db.conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT content_hash
            FROM pages
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    print(f"  Groupes de doublons    : {dup_count}")
    _sep()


def cmd_list_dupes(db: Database, limit: int = 20):
    _sep()
    print("  DOUBLONS DE CONTENU (hash identique)")
    _sep()
    rows = db.conn.execute("""
        SELECT content_hash, COUNT(*) as cnt,
               GROUP_CONCAT(url, '  |  ') as urls
        FROM pages
        WHERE content_hash IS NOT NULL
        GROUP BY content_hash
        HAVING cnt > 1
        ORDER BY cnt DESC
        LIMIT ?
    """, (limit,)).fetchall()
    if not rows:
        print("  Aucun doublon détecté.")
        return
    for row in rows:
        print(f"\n  hash={row['content_hash'][:16]}…  ({row['cnt']} occurrences)")
        for u in (row['urls'] or "").split("  |  ")[:5]:
            print(f"    {u}")
    _sep()


def cmd_remove_dupes(db: Database):
    _sep()
    print("  SUPPRESSION DES DOUBLONS")
    _sep()
    # Garder le plus ancien (id MIN) pour chaque hash
    result = db.conn.execute("""
        DELETE FROM pages
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM pages
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
        )
        AND content_hash IS NOT NULL
    """)
    db.conn.commit()
    print(f"  {result.rowcount} pages dupliquées supprimées.")
    _sep()


def cmd_remove_short(db: Database, min_words: int):
    _sep()
    print(f"  SUPPRESSION PAGES COURTES (< {min_words} mots)")
    _sep()
    result = db.conn.execute(
        "DELETE FROM pages WHERE word_count < ? OR word_count IS NULL",
        (min_words,)
    )
    db.conn.commit()
    print(f"  {result.rowcount} pages supprimées.")
    _sep()


def cmd_remove_nulls(db: Database):
    _sep()
    print("  SUPPRESSION PAGES SANS CONTENU")
    _sep()
    result = db.conn.execute(
        "DELETE FROM pages WHERE content IS NULL OR TRIM(content) = ''"
    )
    db.conn.commit()
    print(f"  {result.rowcount} pages supprimées.")
    _sep()


def cmd_export_jsonl(db: Database, output_path: str):
    _sep()
    print(f"  EXPORT JSONL -> {output_path}")
    _sep()
    count = db.export_jsonl(output_path)
    print(f"  {count} entrées exportées.")
    _sep()


def cmd_vacuum(db: Database):
    _sep()
    print("  VACUUM SQLite")
    _sep()
    before = os.path.getsize(db.db_file) / (1024 * 1024)
    db.conn.execute("VACUUM")
    db.conn.commit()
    after = os.path.getsize(db.db_file) / (1024 * 1024)
    print(f"  Avant : {before:.2f} Mo  ->  Après : {after:.2f} Mo  (gain : {before - after:.2f} Mo)")
    _sep()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nettoyage et audit de la base crawler SQLite"
    )
    parser.add_argument("--db", default=CRAWLER_CONFIG.get("db_file", "data/datasets/crawler.db"),
                        help="Chemin vers la base SQLite")
    parser.add_argument("--jsonl", default=CRAWLER_CONFIG.get("jsonl_output", "data/datasets/dataset.jsonl"),
                        help="Chemin de sortie JSONL")
    parser.add_argument("--stats",         action="store_true", help="Afficher les statistiques")
    parser.add_argument("--dupes",         action="store_true", help="Lister les doublons")
    parser.add_argument("--remove-dupes",  action="store_true", help="Supprimer les doublons")
    parser.add_argument("--remove-short",  type=int, metavar="N", nargs="?", const=50,
                        help="Supprimer pages < N mots (défaut 50)")
    parser.add_argument("--remove-nulls",  action="store_true", help="Supprimer pages sans contenu")
    parser.add_argument("--export-jsonl",  action="store_true", help="Exporter le dataset JSONL")
    parser.add_argument("--vacuum",        action="store_true", help="VACUUM SQLite")
    parser.add_argument("--all",           action="store_true", help="Tout faire")

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"[ERROR] Base de données introuvable : {args.db}")
        sys.exit(1)

    with Database(db_file=args.db) as db:
        do_all = args.all

        if args.stats or do_all:
            cmd_stats(db)

        if args.dupes or do_all:
            cmd_list_dupes(db)

        if args.remove_nulls or do_all:
            cmd_remove_nulls(db)

        if args.remove_short is not None or do_all:
            min_w = args.remove_short if args.remove_short is not None else 50
            cmd_remove_short(db, min_w)

        if args.remove_dupes or do_all:
            cmd_remove_dupes(db)

        if args.export_jsonl or do_all:
            cmd_export_jsonl(db, args.jsonl)

        if args.vacuum or do_all:
            cmd_vacuum(db)

        if do_all:
            # Stats finales après nettoyage
            print("\n  ÉTAT FINAL APRÈS NETTOYAGE")
            cmd_stats(db)


if __name__ == "__main__":
    main()
