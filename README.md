



# Etape du projet de LLM GPT
# Etape 1 : Veille et compréhension des concepts clés
* Comprendre les concepts de base : LLM, GPT, auto‑régression, attention, etc.
* Lire des articles, tutoriels, vidéos sur les modèles de langage et leur fonctionnement.
* Explorer les architectures de modèles de langage modernes (Transformer, GPT, etc.)
* Se familiariser avec les bibliothèques Python recommandées (numpy, torch, etc.)
* Création d'un cahier des charges détaillé pour le projet (objectifs, étapes, ressources, etc.)

# Etape 2 : Collecte de données
* Identifier des sources de données textuelles (corpus, livres, articles, etc.)
* Utiliser des outils de web scraping (BeautifulSoup, Scrapy) pour collecter des données à partir de sites web.
* Nettoyer et prétraiter les données collectées (suppression de balises HTML, normalisation, etc.)
* Stocker les données prétraitées dans un format structuré (JSON, db)
* Créer un script de collecte de données automatisé (crawler) pour faciliter la collecte future.
* néttoyer les données collectées (suppression de balises HTML, normalisation, etc.)
* Stocker les données prétraitées dans un format structuré (JSON, db)
* Créer un script de collecte de données automatisé (crawler) pour faciliter la collecte future.

# Etape 3 : Tokenisation et création du dataset



# ------------------------------------------------------------------------

# Outils externes à l'IA Génératrice
* Outils de web scraping : WebCrawler -> activation : python crawler.py
* Outils de prétraitement de données : nettoyage, tokenisation, etc. -> activation : python clean_db.py
* outils d'ingestion de données local : ingestion de données à partir de fichiers locaux (PDF, TXT, etc.) -> activation : python ingest_local.py | exemple conseiller pour wikipédia FR :
python ingest_local.py --files "file_location" --commit-every 200 --flush-every 200 --progress-every 250 --checkpoint-every 2000


# pipeline de développement
Dataset texte
Tokenisation (BPE)
Construction du dataset
Embedding
Position encoding
Self-attention
Multi-head attention
Transformer block
Modèle GPT
Fonction de perte
Optimisation (AdamW)
Entraînement
Génération de texte
Scaling et optimisation


