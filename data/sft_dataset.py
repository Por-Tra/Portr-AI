#-*- coding: utf-8 -*-
"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @Lucas Contreras Hodapp

File name : sft_dataset.py


description du script:
Dataset PyTorch for padded supervised fine-tuning examples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """ Dataset PyTorch pour les exemples de fine-tuning supervisé avec padding.

    Args:
        Dataset (Object): Classe de base pour les datasets PyTorch.
    """

    def __init__(self, path: str | Path):
        with np.load(path, allow_pickle=False) as data:

            if "input_ids" not in data or "labels" not in data:
                raise ValueError(f"Le fichier SFT doit contenir input_ids et labels: {path}")
            
            input_ids, labels = data["input_ids"], data["labels"]
        data = {"input_ids": input_ids, "labels": labels}
        if "input_ids" not in data or "labels" not in data:
            raise ValueError(f"Le fichier SFT doit contenir input_ids et labels: {path}")
        
        if data["input_ids"].ndim != 2 or data["labels"].ndim != 2:
            raise ValueError("input_ids et labels doivent etre des tableaux 2D")
        
        if data["input_ids"].shape != data["labels"].shape:
            raise ValueError("input_ids et labels doivent avoir la meme forme")
        
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]
        if not np.issubdtype(self.input_ids.dtype, np.integer) or not np.issubdtype(self.labels.dtype, np.integer):
            raise ValueError("Les IDs et labels SFT doivent etre entiers")
        if self.input_ids.shape[1] == 0 or np.any(self.input_ids < 0) or np.any((self.labels < 0) & (self.labels != -100)):
            raise ValueError("IDs, labels ou longueur SFT invalides")
        if np.any((self.labels != -100).sum(axis=1) == 0):
            raise ValueError("Exemple SFT sans token supervise")
        mask = self.labels[:, :-1] != -100
        if np.any(self.labels[:, -1] != -100) or np.any(self.labels[:, :-1][mask] != self.input_ids[:, 1:][mask]):
            raise ValueError("Labels SFT mal alignes: le label doit etre le token suivant")

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int):
        return (torch.from_numpy(self.input_ids[index].astype(np.int64, copy=False)),
            torch.from_numpy(self.labels[index].astype(np.int64, copy=False))
        )
