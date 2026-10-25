<div align="center">

![Portr-AI logo](asset/logo.png)

# Portr-AI

**A French GPT-style language model, built from scratch**

Every component implemented by hand, from raw corpus to text generation.

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-GPT_Transformer-7952B3?style=flat-square)
![Corpus](https://img.shields.io/badge/Corpus-French-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-In_development-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

[English](#english) | [Français](#francais)

</div>

---

<a id="english"></a>

## Overview

**Portr-AI** is an autoregressive, GPT-style language model for French, written from scratch in Python and PyTorch. It uses no pretrained weights and no high-level model library.

The goal is educational and technical: implement every building block of a Transformer language model, from BPE tokenization to GPU training and text generation.

- Complete pipeline: data collection, cleaning, tokenization, training, inference
- Custom BPE tokenizer, optimized with a max-heap
- Decoder-only Transformer with Flash Attention and KV cache
- Resumable training with mixed precision and asynchronous checkpoints

The data sources, tokenizer and quality filters are configured for French. The pipeline can be adapted to another language by changing the data sources and the tokenizer settings.



**Contents:** [Pipeline](#pipeline) · [Model architecture](#model-architecture) · [Installation](#installation) · [Usage](#usage) · [Configuration](#configuration) · [Technical optimizations](#technical-optimizations) · [Project structure](#project-structure) · [Roadmap](#roadmap) · [References](#references)

---

## Pipeline

```mermaid
flowchart LR
    A["Collection<br/>web and local sources"] --> B["Cleaning<br/>chunking, near-dedup"]
    B --> C["Tokenization<br/>BPE from scratch"]
    C --> D["Training<br/>AdamW, AMP, checkpoints"]
    D --> E["Inference<br/>KV cache, top-k sampling"]
```

| Stage | What it does | Output |
|---|---|---|
| Data preparation | Streams Fineweb2-HQ, filters lightly, splits by document hash, trains BPE on a sample, and encodes with EOS | `train.jsonl`, `val.jsonl`, `*.bin` |
| Tokenization | BPE model trained only on the tokenizer sample | `bpe_model.json` |
| Training | Next-token prediction on sliding windows | checkpoints, final model |
| Inference | Autoregressive generation with KV cache | generated text |

---

## Model architecture

```mermaid
flowchart TB
    T["Input tokens<br/>[batch, seq_length]"] --> E["Token + position embedding<br/>dropout"]
    E --> B
    subgraph B["Transformer block, repeated N times"]
        direction TB
        N1["Pre-LayerNorm"] --> A["Causal multi-head attention<br/>Flash Attention, KV cache<br/>+ residual"]
        A --> N2["Pre-LayerNorm"] --> M["MLP with GELU<br/>+ residual"]
    end
    B --> F["Final LayerNorm"]
    F --> O["Output projection<br/>tied with token embedding"]
    O --> L["Logits<br/>[batch, seq_length, vocab_size]"]
```

- Decoder-only GPT with causal self-attention and pre-LayerNorm blocks
- Output head weight-tied to the token embedding
- Flash Attention through `scaled_dot_product_attention`, and a KV cache at decode time
- GPT-2 style scaled initialization

Model sizes are listed in [Configuration](#configuration).

---

## Installation

**Requirements:** Python 3.10+, CUDA (optional but strongly recommended for training), about 6 GB of RAM at minimum and 16 GB or more recommended.

```bash
git clone https://github.com/Por-tra/Portr-AI.git
cd Portr-AI
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

For GPU training, install PyTorch with the official command matching your CUDA version, for example CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Optional PDF and Wikipedia ZIM support:

```bash
pip install pdfminer.six libzim
```

---

## Usage

### 1. Data preparation

The complete data pipeline is in `prepare_data.py`. It first checks the available
Fineweb2-HQ split names, then streams the selected split, trains the BPE tokenizer
on a small sample, separates validation documents by a stable hash of `id`, and
encodes each document with `<EOS>` into memory-mappable `int32` files.

```bash
python prepare_data.py --split fr --token-budget 500000000
```

The generated files are `data/datasets/train.jsonl`, `data/datasets/val.jsonl`,
`artifacts/cache/train_tokens.bin`, and `artifacts/cache/val_tokens.bin`.

### 2. Training

```bash
python main.py
```

The interactive pipeline asks for:

- the tokenizer mode: `auto`, `use_existing` or `retrain`
- the encoding cache mode: `auto`, `use_precomputed` or `recompute`

Training resumes automatically from the last checkpoint when `RESUME_FROM_CHECKPOINT = True`. Curves (loss, perplexity, learning rate) are saved in `artifacts/` and can be viewed with TensorBoard.

<p align="center">
  <img src="asset/courbe.png" width="48%" alt="Training curves, plot 1"/>
  <img src="asset/courbe2.png" width="48%" alt="Training curves, plot 2"/>
</p>

### 3. Inference

Inference runs at the end of `main.py`, or on its own from the saved model by setting `INFERENCE_ONLY_FROM_FINAL_MODEL = True`. An interactive reprompt mode then lets you test several prompts without restarting the pipeline:

```
[main] reprompt mode active
prompt> Tell me a story
[reprompt] tokens: 52
[reprompt] text: Tell me a story about an old castle...

prompt> :quit
```

---

## Configuration

All parameters are centralized in `main.py` unless stated otherwise. Warmup is computed at runtime as `WARMUP_RATIO` of the total number of optimizer steps.

**Model**

| Parameter | Default | Description |
|---|---|---|
| `VOCAB_SIZE` | 16 000 | BPE vocabulary size |
| `EMBEDDING_DIM` | 384 | Embedding dimension |
| `NUM_HEADS` | 8 | Attention heads |
| `NUM_BLOCKS` | 4 | Transformer blocks |
| `SEQ_LENGTH` | 256 | Context length |
| `dropout` | 0.1 | Dropout rate |
| Total parameters | about 18.7M | `small` configuration |

**Training**

| Parameter | Default | Description |
|---|---|---|
| `BATCH_SIZE_CUDA` / `BATCH_SIZE_CPU` | 32 / 8 | Batch size per micro-step |
| `GRAD_ACCUM_STEPS_CUDA` / `GRAD_ACCUM_STEPS_CPU` | 2 / 4 | Effective batch size: 64 on GPU, 32 on CPU |
| `LEARNING_RATE` | 3e-4 | Peak learning rate (cosine decay) |
| `WEIGHT_DECAY` | 0.1 | AdamW, applied to 2D weights only |
| `EPOCHS` | 3 | Number of epochs |
| `WARMUP_RATIO` | 0.02 | Linear warmup, 2% of optimizer steps |
| `LABEL_SMOOTHING` | 0.1 | Reduces overconfidence |
| `Z_LOSS_WEIGHT` | 1e-4 | Logit stabilization |
| `USE_AMP` | True | Mixed precision (float16 on GPU) |
| `CHECKPOINT_EVERY` | 500 | Asynchronous checkpoint frequency, in steps |
| `RESUME_FROM_CHECKPOINT` | True | Resume from the last checkpoint |

**Inference**

| Parameter | Default | Description |
|---|---|---|
| `INFERENCE_ONLY_FROM_FINAL_MODEL` | False | Skip training and load the final model |
| `INFER_PROMPT` | custom | Starting prompt |
| `INFER_MAX_NEW_TOKENS` | 80 | Maximum generated tokens |
| `INFER_TEMPERATURE` | 0.9 | Sampling temperature |
| `INFER_TOP_K` | 20 | Top-k sampling |

**Data preparation** (command-line options)

| Option | Value | Description |
|---|---|---|
| `--split` | `fr` | Split checked against the configuration's available splits |
| `--token-budget` | 500M | Approximate maximum number of encoded tokens |
| `--tokenizer-sample-bytes` | 300 MiB | Raw sample used to train the BPE |
| `--val-percent` | 1 | Hash bucket percentage reserved for validation |

---

## Technical optimizations

| Area | Technique | Benefit |
|---|---|---|
| BPE tokenizer | Incremental pair counting, max-heap with lazy deletion, word-to-ids cache | Fast training and encoding without the naive O(N x V) pair count |
| Attention | Flash Attention (`sdpa`), KV cache, `is_causal` disabled during decode | 2-4x less VRAM, no recomputation of past tokens |
| Training | Mixed precision with `GradScaler`, gradient accumulation, fused AdamW with decay/no-decay groups, cosine schedule with warmup | Stable training and a large effective batch on limited VRAM |
| I/O | Asynchronous checkpoints in a daemon thread, deferred `loss.item()` | No blocking on disk writes or GPU-CPU sync |
| Data | Lightweight filtering, document-hash validation split, EOS-separated streaming encoding, memory-mapped binary cache (`int32`) | Corpus never fully loaded in RAM |

The `is_causal` flag must be `False` during decode (when `past_kv is not None`), otherwise the whole KV cache is masked out and ignored.

---

## Project structure

```
Portr-AI/
├── main.py                  # Pipeline entry point and configuration
├── data/                    # JSONL samples and PyTorch dataset windows
├── tokenizer/               # BPE from scratch and saved model
├── model/                   # Embedding, attention, transformer block, GPT model
├── training/                # Training loop, loss, optimizer
├── inference/               # Text generation with KV cache
├── artifacts/               # Checkpoints, final model, binary token cache
├── asset/                   # README images
└── prepare_data.py          # Fineweb2-HQ streaming, tokenizer, and encoding
```

---

## Roadmap

- [x] BPE tokenizer from scratch (heap-optimized)
- [x] Data pipeline: collection, cleaning, near-deduplication
- [x] GPT model: embeddings, attention with KV cache, Transformer blocks
- [x] Training loop: mixed precision, gradient accumulation, async checkpoints
- [x] Text generation: top-k, temperature, reprompt mode
- [ ] Evaluation: perplexity curves, BLEU, human evaluation
- [ ] Scaling: larger dataset, medium model of about 50M parameters
- [ ] Advanced optimizations: Flash Attention v2, gradient checkpointing, extended KV cache

---

## References

**Foundational papers**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) — GPT-2 paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar

**BPE tokenization**

- [BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — Sebastian Raschka
- [BPE explained (video)](https://youtu.be/VrZTHO1E76w)

---

## Author and license

**Lucas** — personal project to learn and implement a French-language LLM from scratch. Released under the MIT License.

<br/>

---

<a id="francais"></a>

<div align="center">

<h1>Version française</h1>

**Un modèle de langage de type GPT en français, construit from scratch**

Chaque composant est implémenté à la main, du corpus brut à la génération de texte.

[English](#english) | [Français](#francais)

</div>

---

## Présentation

**Portr-AI** est un modèle de langage auto-régressif de type GPT pour le français, écrit from scratch en Python et PyTorch. Il n'utilise ni poids pré-entraînés, ni bibliothèque de modèles de haut niveau.

L'objectif est pédagogique et technique : implémenter chaque brique d'un modèle de langage Transformer, de la tokenisation BPE à l'entraînement sur GPU et à la génération de texte.

- Pipeline complet : collecte, nettoyage, tokenisation, entraînement, inférence
- Tokenizer BPE maison, optimisé avec un max-heap
- Transformer decoder-only avec Flash Attention et KV cache
- Entraînement reprenable, en précision mixte, avec checkpoints asynchrones

Les sources de données, le tokenizer et les filtres de qualité sont configurés pour le français. Le pipeline peut être adapté à une autre langue en modifiant les sources de données et les paramètres du tokenizer.



**Sommaire :** [Chaîne de traitement](#chaîne-de-traitement) · [Architecture du modèle](#architecture-du-modèle) · [Prérequis et installation](#prérequis-et-installation) · [Utilisation](#utilisation) · [Paramètres](#paramètres) · [Optimisations techniques](#optimisations-techniques) · [Structure du projet](#structure-du-projet) · [Feuille de route](#feuille-de-route) · [Références](#références)

---

## Chaîne de traitement

```mermaid
flowchart LR
    A["Préparation<br/>Fineweb2-HQ streaming"] --> B["Tokenisation<br/>BPE sur échantillon"]
    B --> C["Encodage<br/>EOS + cache int32"]
    C --> D["Entraînement<br/>AdamW, AMP, checkpoints"]
    D --> E["Inférence<br/>KV cache, échantillonnage top-k"]
```

| Étape | Rôle | Sortie |
|---|---|---|
| Préparation | Filtre légèrement et sépare train/validation par hash de `id` | `train.jsonl`, `val.jsonl` |
| Tokenisation | Entraîne le modèle BPE sur un échantillon limité | `bpe_model.json` |
| Encodage | Encode en streaming avec `<EOS>` entre les documents | `train_tokens.bin`, `val_tokens.bin` |
| Entraînement | Prédiction du token suivant sur fenêtres glissantes | checkpoints, modèle final |
| Inférence | Génération auto-régressive avec KV cache | texte généré |

---

## Architecture du modèle

```mermaid
flowchart TB
    T["Tokens en entrée<br/>[batch, seq_length]"] --> E["Embedding de tokens + positions<br/>dropout"]
    E --> B
    subgraph B["Bloc Transformer, répété N fois"]
        direction TB
        N1["Pre-LayerNorm"] --> A["Attention multi-têtes causale<br/>Flash Attention, KV cache<br/>+ résiduel"]
        A --> N2["Pre-LayerNorm"] --> M["MLP avec GELU<br/>+ résiduel"]
    end
    B --> F["LayerNorm finale"]
    F --> O["Projection de sortie<br/>liée à l'embedding de tokens"]
    O --> L["Logits<br/>[batch, seq_length, vocab_size]"]
```

- GPT decoder-only avec auto-attention causale et blocs en pre-LayerNorm
- Tête de sortie partageant ses poids avec l'embedding de tokens
- Flash Attention via `scaled_dot_product_attention`, et KV cache au décodage
- Initialisation mise à l'échelle, comme GPT-2

Les tailles du modèle sont indiquées dans [Paramètres](#paramètres).

---

## Prérequis et installation

**Prérequis :** Python 3.10+, CUDA (optionnel mais fortement recommandé pour l'entraînement), environ 6 Go de RAM au minimum et 16 Go ou plus recommandés.

```bash
git clone https://github.com/Por-tra/Portr-AI.git
cd Portr-AI
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/macOS
pip install -r requirements.txt
```

Pour l'entraînement sur GPU, installer PyTorch avec la commande officielle correspondant à votre version de CUDA, par exemple CUDA 12.1 :

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Utilisation

### 1. Préparation des données

`prepare_data.py` vérifie d’abord les splits de `Fineweb2-HQ`, échantillonne le
corpus en streaming, entraîne le BPE sur quelques centaines de Mo, réserve la
validation selon le hash stable de `id`, puis écrit un token `<EOS>` entre les
documents dans les caches `int32`.

```bash
python prepare_data.py --split fr --token-budget 500000000
```

Sorties : `data/datasets/train.jsonl`, `data/datasets/val.jsonl`,
`artifacts/cache/train_tokens.bin` et `artifacts/cache/val_tokens.bin`.

### 2. Entraînement

```bash
python main.py
```

Le pipeline interactif demande :

- le mode du tokenizer : `auto`, `use_existing` ou `retrain`
- le mode du cache d'encodage : `auto`, `use_precomputed` ou `recompute`

L'entraînement reprend automatiquement depuis le dernier checkpoint si `RESUME_FROM_CHECKPOINT = True`. Les courbes (loss, perplexité, learning rate) sont sauvegardées dans `artifacts/` et visualisables avec TensorBoard.

<p align="center">
  <img src="asset/courbe.png" width="48%" alt="Courbes d'entraînement, graphique 1"/>
  <img src="asset/courbe2.png" width="48%" alt="Courbes d'entraînement, graphique 2"/>
</p>

### 3. Inférence

L'inférence s'exécute à la fin de `main.py`, ou seule à partir du modèle sauvegardé en réglant `INFERENCE_ONLY_FROM_FINAL_MODEL = True`. Un mode interactif permet ensuite de tester plusieurs prompts sans relancer le pipeline :

```
[main] reprompt mode active
prompt> Raconte-moi une histoire
[reprompt] tokens: 52
[reprompt] text: Raconte-moi une histoire sur un vieux château...

prompt> :quit
```

---

## Paramètres

Tous les paramètres sont centralisés dans `main.py`, sauf mention contraire. Le warmup est calculé à l'exécution comme `WARMUP_RATIO` du nombre total de pas d'optimisation.

**Modèle**

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `VOCAB_SIZE` | 16 000 | Taille du vocabulaire BPE |
| `EMBEDDING_DIM` | 384 | Dimension des embeddings |
| `NUM_HEADS` | 8 | Têtes d'attention |
| `NUM_BLOCKS` | 4 | Blocs Transformer |
| `SEQ_LENGTH` | 256 | Longueur du contexte |
| `dropout` | 0.1 | Taux de dropout |
| Paramètres totaux | environ 18,7 M | Configuration `small` |

**Entraînement**

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `BATCH_SIZE_CUDA` / `BATCH_SIZE_CPU` | 32 / 8 | Taille de batch par micro-pas |
| `GRAD_ACCUM_STEPS_CUDA` / `GRAD_ACCUM_STEPS_CPU` | 2 / 4 | Batch effectif : 64 sur GPU, 32 sur CPU |
| `LEARNING_RATE` | 3e-4 | Learning rate maximal (décroissance cosinus) |
| `WEIGHT_DECAY` | 0.1 | AdamW, appliqué aux poids 2D uniquement |
| `EPOCHS` | 3 | Nombre d'epochs |
| `WARMUP_RATIO` | 0.02 | Warmup linéaire, 2 % des pas d'optimisation |
| `LABEL_SMOOTHING` | 0.1 | Réduit la sur-confiance |
| `Z_LOSS_WEIGHT` | 1e-4 | Stabilisation des logits |
| `USE_AMP` | True | Précision mixte (float16 sur GPU) |
| `CHECKPOINT_EVERY` | 500 | Fréquence des checkpoints asynchrones, en pas |
| `RESUME_FROM_CHECKPOINT` | True | Reprise depuis le dernier checkpoint |

**Inférence**

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `INFERENCE_ONLY_FROM_FINAL_MODEL` | False | Saute l'entraînement et charge le modèle final |
| `INFER_PROMPT` | personnalisé | Prompt de départ |
| `INFER_MAX_NEW_TOKENS` | 80 | Nombre maximal de tokens générés |
| `INFER_TEMPERATURE` | 0.9 | Température d'échantillonnage |
| `INFER_TOP_K` | 20 | Échantillonnage top-k |

**Préparation des données** (options en ligne de commande)

| Option | Valeur | Description |
|---|---|---|
| `--split` | `fr` | Split vérifié parmi ceux disponibles pour la configuration |
| `--token-budget` | 500M | Budget approximatif de tokens encodés |
| `--tokenizer-sample-bytes` | 300 MiB | Échantillon brut utilisé pour entraîner le BPE |
| `--val-percent` | 1 | Pourcentage de buckets hash réservé à la validation |

---

## Optimisations techniques

| Domaine | Technique | Bénéfice |
|---|---|---|
| Tokenizer BPE | Comptage incrémental des paires, max-heap avec suppression paresseuse, cache mot vers ids | Entraînement et encodage rapides, sans le comptage naïf en O(N x V) |
| Attention | Flash Attention (`sdpa`), KV cache, `is_causal` désactivé au décodage | 2 à 4 fois moins de VRAM, aucun recalcul du passé |
| Entraînement | Précision mixte avec `GradScaler`, accumulation de gradients, AdamW fusionné avec groupes decay/no-decay, schedule cosinus avec warmup | Entraînement stable et grand batch effectif malgré une VRAM limitée |
| E/S | Checkpoints asynchrones dans un thread daemon, `loss.item()` différé | Pas de blocage sur les écritures disque ni de synchronisation GPU-CPU |
| Données | Filtre léger, split validation par hash de document, EOS entre documents, encodage en streaming, cache binaire mémoire-mappé (`int32`) | Corpus jamais chargé entièrement en RAM |

Le flag `is_causal` doit être à `False` au décodage (quand `past_kv is not None`), sinon tout le KV cache est masqué et ignoré.

---

## Structure du projet

```
Portr-AI/
├── main.py                  # Point d'entrée du pipeline et configuration
├── data/                    # JSONL et fenêtres glissantes PyTorch
├── tokenizer/               # BPE from scratch et modèle sauvegardé
├── model/                   # Embedding, attention, bloc Transformer, modèle GPT
├── training/                # Boucle d'entraînement, loss, optimiseur
├── inference/               # Génération de texte avec KV cache
├── artifacts/               # Checkpoints, modèle final, cache binaire de tokens
├── asset/                   # Images du README
└── prepare_data.py          # Fineweb2-HQ, tokenizer et encodage
```

---

## Feuille de route

- [x] Tokenizer BPE from scratch (optimisé avec un heap)
- [x] Pipeline de données : collecte, nettoyage, quasi-dédoublonnage
- [x] Modèle GPT : embeddings, attention avec KV cache, blocs Transformer
- [x] Boucle d'entraînement : précision mixte, accumulation de gradients, checkpoints asynchrones
- [x] Génération de texte : top-k, température, mode reprompt
- [ ] Évaluation : courbes de perplexité, BLEU, évaluation humaine
- [ ] Passage à l'échelle : dataset plus grand, modèle moyen d'environ 50 M de paramètres
- [ ] Optimisations avancées : Flash Attention v2, gradient checkpointing, KV cache étendu

---

## Références

**Articles fondateurs**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) — article GPT-2
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar

**Tokenisation BPE**

- [BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — Sebastian Raschka
- [BPE expliqué (vidéo)](https://youtu.be/VrZTHO1E76w)

---

## Auteur et licence

**Lucas** — projet personnel d'apprentissage et d'implémentation from scratch d'un LLM en français. Publié sous licence MIT.