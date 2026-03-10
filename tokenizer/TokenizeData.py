"""
TokenizeData.py

Fichier à exécuter pour tokeniser les données brutes récupérées par le crawler.


"""

from tokenizer import Tokenizer
import json
import sys


def _configure_stdout_for_unicode() -> None:
    """Force UTF-8 output on compatible terminals to avoid cp1252 crashes."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        # Some environments do not support reconfigure; keep default stream.
        pass


def TokenizeData():

    # Vérifier si le fichier de modèle existe déjà pour éviter de réentraîner à chaque exécution
    try:
        tok = Tokenizer.load()
        print("Tokenizer model loaded from file.")
        return
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
        target_vocab_size=5000,
        min_pair_freq=2,
        verbose=True
    )

    tok.save()

    print("Tokenizer saved")

if __name__ == "__main__":
    TokenizeData()