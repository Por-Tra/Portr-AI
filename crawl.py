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

# ── Fix Windows : le ProactorEventLoop (défaut sur Python 3.8+/Windows) lève
# des ConnectionResetError / WinError 10054 sur les connexions fermées par le
# serveur distant. SelectorEventLoop ne présente pas ce bug avec aiohttp.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Silence les tracebacks asyncio internes résiduels (ex: _call_connection_lost)
    import logging
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

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
    default_urls = config.get("start_urls", [])
    parser = argparse.ArgumentParser(
        description="Crawler web récursif — collecte de texte vers SQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemples :\n"
               "  python crawl.py\n"
               "  python crawl.py --url https://example.com\n"
               "  python crawl.py --url https://site1.com https://site2.com\n",
    )
    parser.add_argument(
        "--url",
        nargs="+",
        default=None,
        metavar="URL",
        help="URL(s) de départ (écrase start_urls de config.py)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=config["max_pages"],
        help=f"Limite globale de pages (défaut : {config['max_pages']})",
    )
    parser.add_argument(
        "--max-pages-per-site",
        type=int,
        default=config.get("max_pages_per_site", config["max_pages"]),
        help="Limite de pages par site avant de passer au suivant",
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

    config = dict(config)
    config["start_urls"] = args.url if args.url else default_urls
    config["max_pages"] = args.max_pages
    config["max_pages_per_site"] = args.max_pages_per_site
    config["max_depth"] = args.max_depth
    config["num_workers"] = args.workers
    config["db_file"] = args.db
    return config


async def main():
    config = parse_args(CRAWLER_CONFIG)
    log = build_logger()

    start_urls = config["start_urls"]

    log("=" * 70, "INFO")
    log("  CRAWLER WEB — démarrage", "INFO")
    log("=" * 70, "INFO")
    log(f"  Sites ({len(start_urls)}) :", "INFO")
    for u in start_urls:
        log(f"    {u}", "INFO")
    log(f"  Pages max (global)   : {config['max_pages']}", "INFO")
    log(f"  Pages max / site     : {config.get('max_pages_per_site', '—')}", "INFO")
    log(f"  Profondeur max       : {config['max_depth']}", "INFO")
    log(f"  Workers              : {config['num_workers']}", "INFO")
    log(f"  Proxies              : {len(config.get('proxies', []))}", "INFO")
    log(f"  User-agents          : {len(config.get('user_agents', []))}", "INFO")
    log(f"  Base de données      : {config['db_file']}", "INFO")
    log("=" * 70, "INFO")

    with Database(db_file=config["db_file"]) as db:
        crawler = WebCrawler(database=db, config=config, logger=log)
        try:
            await crawler.run_all(start_urls)
        except KeyboardInterrupt:
            log("\n[INFO] Interruption clavier — arrêt en cours…", "WARN")
            crawler.stop()

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