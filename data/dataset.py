# -*- coding: utf-8 -*-

"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : dataset.py

description du script:
    Fichier définissant la classe TextDataset, un wrapper PyTorch pour les données textuelles tokenisées.

    Exemple du fonctionnement (seq_length=3, stride=3):
    Tokens : [10,25,33,48,77,90,12]

    dataset:
    input  : [10,25,33]
    target : [25,33,48]

    input  : [48,77,90]
    target : [77,90,12]

    stride=seq_length (défaut) : fenêtres disjointes, chaque token est vu
    une fois par epoch. stride=1 revenait à voir chaque token seq_length
    fois par epoch (256x plus de steps pour quasiment la même information).

    Uitlisation:
    tokens = tokenizer.encode(corpus)
    dataset = TextDataset(tokens, seq_length=128)
    x, y = dataset[0]
"""


import torch
from torch.utils.data import Dataset, Sampler


class TextDataset(Dataset):

    def __init__(self, tokens, seq_length: int, stride: int | None = None):
        self.tokens = tokens
        self.seq_length = seq_length
        self.stride = stride or seq_length

    def __len__(self):
        #& start max = N - seq_length - 1 (la cible déborde d'un token)
        usable = len(self.tokens) - self.seq_length - 1
        if usable < 0:
            return 0
        return usable // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.tokens[start:start + self.seq_length]
        y = self.tokens[start + 1:start + self.seq_length + 1]

        #& Si tokens est deja un Tensor (y compris memory-mapped), evite
        #& une conversion Python intermediaire qui coute en RAM.
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return x.to(torch.long), y.to(torch.long)

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class ResumableRandomSampler(Sampler[int]):
    """
    Sampler aléatoire reproductible : l'ordre dépend uniquement de (seed, epoch).

    Permet une reprise exacte en milieu d'epoch : set_epoch(epoch, start_index)
    régénère la même permutation et saute directement les échantillons déjà vus,
    sans les charger.

    replacement=True : tirage avec remise par blocs (RAM constante), utile si
    le dataset est énorme.
    """

    _CHUNK = 65_536

    def __init__(self, num_samples: int, seed: int = 0, replacement: bool = False):
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.replacement = replacement
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        start = self.start_index
        self.start_index = 0  #& la reprise ne s'applique qu'une fois

        if not self.replacement:
            perm = torch.randperm(self.num_samples, generator=g)
            yield from perm[start:].tolist()
            return

        produced = 0
        while produced < self.num_samples:
            n = min(self._CHUNK, self.num_samples - produced)
            chunk = torch.randint(self.num_samples, (n,), generator=g)
            lo = max(0, start - produced)
            produced += n
            if lo < n:
                yield from chunk[lo:].tolist()
