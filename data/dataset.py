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

        # Si tokens est deja un Tensor (y compris memory-mapped), evite
        # une conversion Python intermediaire qui coute en RAM.
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.dtype != torch.long:
                x = x.to(torch.long)
                y = y.to(torch.long)
            return x, y

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


