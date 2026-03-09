# -*- coding: utf-8 -*-
"""
Configuration du crawler web.
Modifier ce fichier pour paramétrer le crawl, puis lancer :
    python crawl.py
"""

CRAWLER_CONFIG = {
    # --- Sites à crawler (parcourus dans l'ordre) ---
    # Dès qu'un site est épuisé (ou que max_pages_per_site est atteint), on passe au suivant.
    "start_urls": [
        "https://fr.wikipedia.org/wiki/Accueil",
        "https://www.lemonde.fr",
        "https://www.futura-sciences.com",
        "https://diply.com/",
        "https://www.bfmtv.com/",
        "https://www.cnews.fr/",
        "https://www.kaggle.com/"
    ],

    # --- Limites ---
    "max_pages": 10_000,          # Limite GLOBALE (tous sites confondus)
    "max_pages_per_site": 2_000,  # Limite PAR SITE avant de passer au suivant
    "max_depth": 10,              # Profondeur de récursion max (0 = page de départ seule)

    # --- Parallélisme ---
    "num_workers": 8,             # Nombre de workers asynchrones par site
    "concurrency": 50,            # Nb max de connexions TCP simultanées

    # --- Délai de politesse ---
    "politeness_delay": 0.3,      # Délai (secondes) entre deux requêtes par worker

    # --- Rotation User-Agent ---
    # Laisser vide [] pour utiliser les UA intégrés par défaut.
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    ],

    # --- Rotation de proxies ---
    # Format : "http://user:pass@host:port"  ou  "http://host:port"
    # Laisser vide [] pour connexion directe (sans proxy).
    "proxies": [
        # "http://proxy1.example.com:8080",
        # "http://user:pass@proxy2.example.com:3128",
    ],
    # Durée (secondes) pendant laquelle un proxy pénalisé (403/429) est mis de côté
    "proxy_cooldown": 60.0,

    # --- Filtre qualité du texte ---
    "min_words": 10,              # Ignorer les pages avec moins de N mots
    "max_words": 100_000,         # Ignorer les pages avec plus de N mots (dumps géants)
    "min_letters_ratio": 0.70,    # Proportion min de caractères alphabétiques (0.0 – 1.0)

    # --- Filtre de langue ---
    # Codes ISO 639-1. Laisser vide [] pour tout accepter.
    "target_langs": ["fr"],

    # --- Filtre URL ---
    # Pages dont l'URL contient un de ces tokens sont ignorées.
    "url_blacklist": [
        "login", "signin", "signup", "register",
        "logout", "account", "password",
        "privacy", "cookies", "terms", "legal",
        "cart", "checkout", "panier",
    ],

    # --- Sorties ---
    "db_file": "data/datasets/crawler.db",
    "jsonl_output": "data/datasets/dataset.jsonl",
}
