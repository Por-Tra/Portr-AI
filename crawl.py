# -*- coding: utf-8 -*-
"""
Point d'entrée CLI du crawler web.

Usage :
    python crawl.py
    python crawl.py --url https://example.com
    python crawl.py --url https://example.com --max-pages 100 --max-depth 3

Les valeurs passées en argument écrasent celles de Crawler/config.py.
"""

import asyncio
import argparse
import sys
import os

# Assure que le dossier racine est dans le path
sys.path.insert(0, os.path.dirname(__file__))

from Crawler.config import CRAWLER_CONFIG
from Crawler.webcrawler import WebCrawler
from data.DataBase.database import Database


def build_logger():
    """Logger colorisé simple (pas de dépendance externe)."""
    COLORS = {
        "INFO":    "\033[0m",
        "WARN":    "\033[93m",
        "ERROR":   "\033[91m",
        "SUCCESS": "\033[92m",
    }
    RESET = "\033[0m"

    def log(message: str, level: str = "INFO"):
        color = COLORS.get(level, COLORS["INFO"])
        print(f"{color}{message}{RESET}", flush=True)

    return log


def parse_args(config: dict) -> dict:
    parser = argparse.ArgumentParser(
        description="Crawler web récursif — collecte de texte vers SQLite"
    )
    parser.add_argument(
        "--url",
        default=config["start_url"],
        help=f"URL de départ (défaut : {config['start_url']})",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=config["max_pages"],
        help=f"Limite de pages (défaut : {config['max_pages']})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=config["max_depth"],
        help=f"Profondeur max (défaut : {config['max_depth']})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=config["num_workers"],
        help=f"Nombre de workers (défaut : {config['num_workers']})",
    )
    parser.add_argument(
        "--db",
        default=config["db_file"],
        help=f"Chemin vers la base SQLite (défaut : {config['db_file']})",
    )

    args = parser.parse_args()

    # Fusionne les overrides CLI dans la config
    config = dict(config)
    config["start_url"] = args.url
    config["max_pages"] = args.max_pages
    config["max_depth"] = args.max_depth
    config["num_workers"] = args.workers
    config["db_file"] = args.db
    return config


async def main():
    config = parse_args(CRAWLER_CONFIG)
    log = build_logger()

    log("=" * 70, "INFO")
    log("  CRAWLER WEB — démarrage", "INFO")
    log("=" * 70, "INFO")
    log(f"  URL de départ   : {config['start_url']}", "INFO")
    log(f"  Pages max       : {config['max_pages']}", "INFO")
    log(f"  Profondeur max  : {config['max_depth']}", "INFO")
    log(f"  Workers         : {config['num_workers']}", "INFO")
    log(f"  Base de données : {config['db_file']}", "INFO")
    log("=" * 70, "INFO")

    with Database(db_file=config["db_file"]) as db:
        crawler = WebCrawler(database=db, config=config, logger=log)
        try:
            await crawler.run(config["start_url"])
        except KeyboardInterrupt:
            log("\n[INFO] Interruption clavier — arrêt en cours…", "WARN")
            crawler.stop()

        # Statistiques DB
        stats = db.get_statistics()
        if stats:
            log("\n  BASE DE DONNÉES", "INFO")
            log(f"  Pages totales   : {stats['total_pages']}", "INFO")
            log(f"  Domaines uniq.  : {stats['unique_domains']}", "INFO")
            log(f"  Moy. mots/page  : {stats['avg_words']}", "INFO")
            log(f"  Max mots/page   : {stats['max_words']}", "INFO")
            log(f"  Liens totaux    : {stats['total_links']}", "INFO")


if __name__ == "__main__":
    asyncio.run(main())
