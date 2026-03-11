"""
dataset.py

Fichier définissant la classe TextDataset, un wrapper PyTorch pour les données textuelles tokenisées.

Exemple du fonctionnement:
Tokens : [10,25,33,48,77,90,12]

dataset:
input  : [10,25,33,48]
target : [25,33,48,77]

input  : [25,33,48,77]
target : [33,48,77,90]

input  : [33,48,77,90]
target : [48,77,90,12]

Uitlisation:
tokens = tokenizer.encode(corpus)
dataset = TextDataset(tokens, seq_length=128)
x, y = dataset[0]
"""

import torch
from torch.utils.data import Dataset
from tokenizer.tokenizer import Tokenizer

class TextDataset(Dataset):

    def __init__(self, tokens, seq_length):
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):

        x = self.tokens[idx:idx+self.seq_length]
        y = self.tokens[idx+1:idx+self.seq_length+1]

        return torch.tensor(x), torch.tensor(y)


