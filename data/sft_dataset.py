#-*- coding: utf-8 -*-
"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

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
        data = np.load(path)
        if "input_ids" not in data or "labels" not in data:
            raise ValueError(f"Le fichier SFT doit contenir input_ids et labels: {path}")
        
        if data["input_ids"].ndim != 2 or data["labels"].ndim != 2:
            raise ValueError("input_ids et labels doivent etre des tableaux 2D")
        
        if data["input_ids"].shape != data["labels"].shape:
            raise ValueError("input_ids et labels doivent avoir la meme forme")
        
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.input_ids[index].astype(np.int64, copy=False)),
            torch.from_numpy(self.labels[index].astype(np.int64, copy=False))
        )
