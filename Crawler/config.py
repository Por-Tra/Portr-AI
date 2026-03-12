# -*- coding: utf-8 -*-
"""
config.py
"""

CRAWLER_CONFIG = {
    "start_urls": [
        "https://crunchyclone00.github.io/NovelNeko/index.html",
        "http://mumm.hautetfort.com/tres_long_texte/",
        "https://www.formations-analytics.com/20-ensembles-de-donnees-open-source-pour-lia-generative-et-agentique/",
        "https://www.limpida.com/blog/ia-generative-data-analytics",
        "https://www.laas.fr/documents/415/th%C3%A8se_Qualit%C3%A9_datasetsIA_g%C3%A9n%C3%A9rative.pdf",
        "https://www.innovatiana.com/fr/craft-a-dataset/ai-training",
        "https://www.databricks.com/fr/discover/generative-ai"


    ],
    "stay_on_domain": False,
    "extract_pdf": True,
    "max_pages": 100_000,
    "max_pages_per_site": 1_000,
    "max_depth": 100,
    "num_workers": 20,
    "concurrency": 50,
    "politeness_delay": 0.5,
    "jitter_ratio": 0.5,
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
    ],
    "proxies": [],
    "proxy_cooldown": 60.0,
    "min_words": 10,
    "max_words": 1_000_000,
    "min_letters_ratio": 0.70,
    "target_langs": ["fr"],
    # Note : .pdf retiré de la blacklist — géré par extract_pdf=True/False
    "url_blacklist": [
        "login", "signin", "signup", "register",
        "logout", "account", "password",
        "privacy", "cookies", "terms", "legal",
        "cart", "checkout", "panier",
        ".jpg", ".jpeg", ".png", ".gif", ".webp",
        ".zip", ".exe", ".mp3", ".mp4", ".avi",
        ".css", ".js",
    ],
    "db_file": "data/datasets/crawler.db",
    "jsonl_output": "data/datasets/dataset.jsonl",
}