# -*- coding: utf-8 -*-
"""
Configuration du crawler web.
Modifier ce fichier pour paramétrer le crawl, puis lancer :
    python crawl.py
"""

CRAWLER_CONFIG = {
    # --- URL de départ ---
    "start_url": "https://thunderbit.com/fr/blog/best-free-website-crawler",

    # --- Limites ---
    "max_pages": 5000,       # Nombre maximum de pages à visiter
    "max_depth": 10,         # Profondeur maximale de récursion (0 = page de départ seulement)

    # --- Parallélisme ---
    "num_workers": 8,        # Nombre de workers asynchrones parallèles
    "concurrency": 50,       # Nb max de connexions TCP simultanées (DNS cache inclus)

    # --- Délai de politesse ---
    "politeness_delay": 0.3, # Délai (secondes) entre deux requêtes par worker

    # --- HTTP ---
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0.0.0 Safari/537.36",

    # --- Filtre qualité du texte ---
    "min_words": 50,         # Ignorer les pages avec moins de N mots
    "max_words": 100_000,    # Ignorer les pages avec plus de N mots (dumps géants)
    "min_letters_ratio": 0.70,  # Proportion min de caractères alphabétiques (0.0 – 1.0)

    # --- Filtre de langue ---
    # Liste des codes langue acceptés (ISO 639-1). Laisser vide [] pour tout accepter.
    # Exemples : ["fr"], ["en"], ["fr", "en"]
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
