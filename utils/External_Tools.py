"""

Fichier pour importer les outils externes et les bibliothèques nécessaires pour le projet.
    - numpy → calcul matriciel
    - torch → autograd → GPU → tensor operations
    - tokenizers (optionnel) → tokenisation rapide
    - matplotlib → visualisation
    - tqdm → progression entraînement
    - datasets (optionnel) → charger corpus
    - Important : Ne pas utiliser directement des modèles LLM pré‑entraînés.

"""

# tokenizers (optionnel)
# N/A

#datasets (optionnel)
# N/A

print("="*50)

# Verification de la disponibilité des bibliothèques et importation des bibliothèques nécessaires
try:
    import numpy as np
    print("NumPy is available.")
except ImportError:
    print("NumPy is not available.")

try:
    import torch
    import torch.nn as nn
    print("PyTorch is available.")
except ImportError:
    print("PyTorch is not available.")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    print("Matplotlib is available.")
except ImportError:
    print("Matplotlib is not available.")

try:
    import tqdm
    from tqdm import tqdm
    import tqdm as tqdm_module
    print("tqdm is available.")
except ImportError:
    print("tqdm is not available.")


print("="*50)
print("Device information:")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("="*50)