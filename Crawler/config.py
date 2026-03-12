# -*- coding: utf-8 -*-
"""
config.py

Configuration du crawler web.
Modifier ce fichier pour paramétrer le crawl, puis lancer :
    python crawl.py
"""

CRAWLER_CONFIG = {
    # --- Sites à crawler ---
    # En mode stay_on_domain=True  → crawlés séquentiellement (1 BFS par site)
    # En mode stay_on_domain=False → tous seedés dans un seul BFS global
    "start_urls": [
        "http://mumm.hautetfort.com/tres_long_texte/"
    ],

    # --- Mode de crawl ---
    # True  → crawl intra-site uniquement (comportement original)
    # False → suit tous les liens externes = explore tout internet
    "stay_on_domain": False,

    # --- Limites ---
    "max_pages": 100000,         # Limite GLOBALE (tous sites confondus)
    "max_pages_per_site": 1_000,  # Limite PAR SITE (ignorée si stay_on_domain=False)
    "max_depth": 100,              # Profondeur de récursion max

    # --- Parallélisme ---
    "num_workers": 20,            # Nombre de workers asynchrones
    "concurrency": 50,            # Nb max de connexions TCP simultanées

    # --- Délai de politesse (par domaine) ---
    # Délai minimum entre deux requêtes vers le même domaine.
    # Un jitter aléatoire de ±(politeness_delay × jitter_ratio) est ajouté.
    "politeness_delay": 0.5,      # secondes (augmenté pour éviter les blocages)
    "jitter_ratio": 0.5,          # 0.5 = ±50 % de jitter

    # --- Rotation User-Agent ---
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    ],

    # --- Rotation de proxies ---
    # Format : "http://user:pass@host:port"  ou  "http://host:port"
    # Laisser vide [] pour connexion directe.
    "proxies": [
        # "http://proxy1.example.com:8080",
    ],
    "proxy_cooldown": 60.0,       # Durée (s) de mise en quarantaine d'un proxy bloqué

    # --- Filtre qualité du texte ---
    "min_words": 10,
    "max_words": 100_000,
    "min_letters_ratio": 0.70,

    # --- Filtre de langue ---
    # Codes ISO 639-1. Laisser vide [] pour tout accepter.
    "target_langs": ["fr"],

    # --- Filtre URL ---
    "url_blacklist": [
        "login", "signin", "signup", "register",
        "logout", "account", "password",
        "privacy", "cookies", "terms", "legal",
        "cart", "checkout", "panier",
        ".pdf", ".jpg", ".jpeg", ".png", ".gif",
        ".zip", ".exe", ".mp3", ".mp4",
    ],

    # --- Sorties ---
    "db_file": "data/datasets/crawler.db",
    "jsonl_output": "data/datasets/dataset.jsonl",
}