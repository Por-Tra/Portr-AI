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
import model.embedding as embedding


# Variable de configuration des données et du modèle
td.VOCAB_SIZE = 50_000 # changer ça pour ajuster la taille du vocablaire

# Variable de configuration des sequences
SEQ_LENGTH = 128
BATCH_SIZE = 32


def main():

    print("Starting main process...")

    # Étape 1 : Tokenisation des données
    print("Tokenizing data...")
    td.TokenizeData()

    # Étape 2 : Création du dataset
    # charger les tokens depuis le tokenizer
    print("Loading tokenizer model...")


    try:
        tok = td.Tokenizer.load()
    except FileNotFoundError:
        print("[main.py]Erreur lors de l'ouverture du chargement des données")
        return
    
    tokens = tok.encode_corpus("data/datasets/dataset.jsonl")

    # Étape 3 : Création du dataset pour l'entraînement du modèle

    print("Creating dataset...")

    dataset = TextDataset(tokens, SEQ_LENGTH)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("Dataset size:", len(dataset))

    # Affichage test
    print("Test dataset:")
    print("-" * 20)
    print("Example sample")

    x, y = dataset[0]

    print("input :", x)
    print("target:", y)

    print("input shape :", x.shape)
    print("target shape:", y.shape)
    print("-" * 20)
    print("Test dataloader:")
    batch_x, batch_y = next(iter(dataloader))

    print("Batch input shape :", batch_x.shape)
    print("Batch target shape:", batch_y.shape)
    print("-" * 20)


    # Étape 4 : Entraînement du modèle
#     model
#    ├ embedding
#    ├ transformer
#    └ head

    print("Training model...")

    # Première couche du model : embedding
    print("---"*20)
    print("Embedding layer...")
    print("Initializing embedding layer...")
    embedding = embedding.Embedding(td.VOCAB_SIZE, SEQ_LENGTH, 256)

    print("Testing embedding layer with a batch of data...")
    batch_x, _ = next(iter(dataloader))

    print("Batch input shape before embedding:", batch_x.shape)
    out = embedding(batch_x)

    print(out.shape)




    # Deuxième couche du model : transformer





    # Troisième couche du model : head (couche de sortie)





if __name__ == "__main__":
    main()






