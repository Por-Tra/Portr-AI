# Lancer Portr-AI 1.0.1

La [release 1.0.1](https://github.com/Por-Tra/Portr-AI/releases/tag/v1.0.1)
fournit deux archives autonomes. Choisir **SFT** pour les questions/instructions,
ou **base** pour la continuation de texte. Chaque archive contient ses poids,
le tokenizer compatible et le code d'inférence existant, avec un petit lanceur.
Pas besoin de cloner le dépôt, de récupérer les datasets ou de relancer un entraînement.
Les poids SFT sont complets : le modèle de base n'est pas nécessaire pour les utiliser.

Prérequis : **Python 3.10+**. CPU pris en charge ; CUDA est utilisé automatiquement
si la version installée de PyTorch le permet. L'installation télécharge PyTorch
et tqdm, puis l'inférence fonctionne hors ligne.

## Modèle SFT (recommandé)

Télécharger [portr-ai-1.0.1-sft.zip](https://github.com/Por-Tra/Portr-AI/releases/download/v1.0.1/portr-ai-1.0.1-sft.zip)
et l'extraire, ou utiliser ces commandes (PowerShell, Linux ou macOS) :

```bash
curl -L -o portr-ai-1.0.1-sft.zip https://github.com/Por-Tra/Portr-AI/releases/download/v1.0.1/portr-ai-1.0.1-sft.zip
python -m zipfile -e portr-ai-1.0.1-sft.zip .
cd portr-ai-1.0.1-sft
python -m pip install -r requirements-inference.txt
python run_model.py --prompt "Explique simplement ce qu'est une étoile."
```

Dans Windows PowerShell 5, utiliser `curl.exe` à la place de `curl`.
Un environnement virtuel Python est conseillé si vous n'en utilisez pas déjà un.

Pour poser plusieurs questions indépendantes :

```bash
python run_model.py
```

Entrer `:quit` pour quitter. Le lanceur utilise le format d'instruction SFT
et s'arrête sur le token de fin de réponse. Il ne conserve pas d'historique.
Par défaut, le SFT utilise un décodage déterministe (`--temperature 0`).
Exemple pour forcer le CPU et limiter la génération :

```bash
python run_model.py --device cpu --max-new-tokens 64 --prompt "Bonjour !"
```

## Modèle de base

Télécharger [portr-ai-1.0.1-base.zip](https://github.com/Por-Tra/Portr-AI/releases/download/v1.0.1/portr-ai-1.0.1-base.zip),
l'extraire et ouvrir le dossier `portr-ai-1.0.1-base` :

```bash
python -m pip install -r requirements-inference.txt
python run_model.py --model base --prompt "Il était une fois"
```

La base continue le texte brut, avec une température de 0.9 par défaut.
Les deux variantes acceptent `--temperature` et `--max-new-tokens`.

## Contenu publié

| Variante | Vocabulaire | Contexte | Poids |
|---|---|---|---|
| Base | 16 000 tokens | 256 tokens | `artifacts/models/gpt_final_state.pt` |
| SFT | 16 000 tokens | 1 024 tokens | `artifacts/sft/models/sft_final_state.pt` |

Les deux modèles ont 4 blocs, 8 têtes et des embeddings de dimension 384.
Le SFT publié correspond au run actif `20261123T144022Z-d7212f67`, sélectionné
sur la meilleure cross-entropy de validation (2.5559, epoch 2, étape 1588).
Ce sont de petits modèles expérimentaux : la qualité des réponses reste limitée.

Les archives excluent les datasets, caches, checkpoints d'optimiseur et outils
d'entraînement. `SHA256SUMS.txt` permet de vérifier les téléchargements ;
`manifest.json` dans chaque archive décrit les poids et leurs empreintes.
Les sources complètes restent disponibles séparément sur GitHub.
