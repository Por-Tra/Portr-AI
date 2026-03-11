"""
main.py

!! Lancer ce fichier pour exécuter tout le processus du model.

pipeline principal du projet, orchestrant le processus de tokenisation, de création du dataset et d'entraînement du modèle.
crawler
↓
corpus texte
↓
tokenizer
↓
tokens
↓
dataset GPT  ← ÉTAPE 3
↓
embedding
↓
transformer
↓
training

Important
======================================================================
| On ne met pas le crawler ici, se sera la seul exception, le crawler est un processus à part qui doit être exécuté manuellement pour récupérer les données brutes. 
| Le reste du pipeline (tokenisation, création du dataset, entraînement du modèle) peut être automatisé et orchestré à partir de ce fichier main.py.
|
|
|
|
"""


import data.dataset as ds
import tokenizer.TokenizeData as td
from torch.utils.data import DataLoader
from data.dataset import TextDataset


# Variable de configuration des données et du modèle
td.VOCAB_SIZE = td.VOCAB_SIZE # changer ça pour ajuster la taille du vocablaire

# Variable de configuration des sequences
SEQ_LENGTH = 128
BATCH_SIZE = 32


def main():

    # Étape 1 : Tokenisation des données
    td.TokenizeData()

    # Étape 2 : Création du dataset
    # charger les tokens depuis le tokenizer
    try:
        tok = td.Tokenizer.load()
    except FileNotFoundError:
        print("[main.py]Erreur lors de l'ouverture du chargement des données")
        return
    
    tokens = tok.encode_corpus("data/datasets/dataset.jsonl")

    # Étape 3 : Création du dataset GPT

    dataset = TextDataset(tokens, SEQ_LENGTH)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("Dataset size:", len(dataset))











