"""
TokenizeData.py

fichier qui sert au lancement et la configuration du processus de tokenisation des données textuelles brutes. 
Il utilise la classe Tokenizer pour entraîner un modèle de tokenisation sur les données textuelles extraites par le crawler, 
puis sauvegarde ce modèle pour une utilisation ultérieure dans le pipeline d'entraînement du modèle GPT.

"""


from tokenizer.tokenizer import Tokenizer
import json
import sys

VOCAB_SIZE = 500000


def _configure_stdout_for_unicode() -> None:
    """Force UTF-8 output on compatible terminals to avoid cp1252 crashes."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Some environments do not support reconfigure; keep default stream.
        pass


def TokenizeData():
    _configure_stdout_for_unicode()

    # Vérifier si le fichier de modèle existe déjà pour éviter de réentraîner à chaque exécution
    try:
        tok = Tokenizer.load()
        print("Tokenizer model loaded from file.")
    except FileNotFoundError:
        print("No existing tokenizer model found. Training a new one...")
        tok = Tokenizer()

    texts = []

    with open("data/datasets/dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            texts.append(text)

    print("Training tokenizer...")

    tok.train(
        texts,
        target_vocab_size=VOCAB_SIZE, # plus le vocabulaire est grand, plus le modèle peut comprendre de nuances, mais cela rend aussi le modèle plus lourd et plus lent à entraîner. 100k est un bon compromis pour un modèle de taille moyenne.
        min_pair_freq=2, # plus la fréquence minimale est élevée, moins de paires seront fusionnées, ce qui peut conduire à un vocabulaire plus grand et à une meilleure capacité de généralisation, mais aussi à un modèle plus lourd. 2 est un bon compromis pour éviter les paires très rares qui pourraient ne pas être utiles.
        verbose=True
    )

    tok.save()

    print("Tokenizer saved")
