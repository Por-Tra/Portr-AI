<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-En_développement-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Langue-Français-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA5MCA2MCI+PHJlY3Qgd2lkdGg9IjkwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjRUQyOTM5Ii8+PHJlY3Qgd2lkdGg9IjYwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjZmZmIi8+PHJlY3Qgd2lkdGg9IjMwIiBoZWlnaHQ9IjYwIiBmaWxsPSIjMDAyMzk1Ii8+PC9zdmc+"/>
<img src="https://img.shields.io/badge/Architecture-Transformer_GPT-7952B3?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

```
██████╗  ██████╗ ██████╗ ████████╗██████╗       █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗     ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝
```

### **Un LLM GPT français, implémenté from scratch**
*Comprendre chaque rouage — du corpus brut à la génération de texte*

<br/>

[![📖 Documentation Wiki](https://img.shields.io/badge/📖_Documentation_Wiki-por--tra.github.io-4A90D9?style=for-the-badge)](https://por-tra.github.io/Portr-AI-DOC/)

</div>

---

### Note importante ! 
> Le projet n'a pas pu être tester correctement et à 100% sur le dataset complet à cause de crontraintes matérielles (VRAM GPU, temps d'entraînement).

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Architecture](#-architecture)
- [Pipeline complet](#-pipeline-complet)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
  - [1. Collecte de données](#1-collecte-de-données)
  - [2. Nettoyage et prétraitement](#2-nettoyage-et-prétraitement)
  - [3. Entraînement](#3-entraînement)
  - [4. Inférence](#4-inférence)
- [Hyperparamètres](#-hyperparamètres)
- [Optimisations techniques](#-optimisations-techniques)
- [Roadmap](#-roadmap)
- [Ressources](#-ressources)

---

## 🧠 Vue d'ensemble

**Portr-AI** est un modèle de langage auto-régressif de type GPT, construit entièrement **from scratch en Python/PyTorch**, sans utilisation d'aucun modèle pré-entraîné.

L'objectif est pédagogique et technique : **maîtriser chaque brique** de l'architecture Transformer, de la tokenisation BPE jusqu'à la génération de texte, en passant par l'entraînement sur GPU.

> 💡 Le modèle est orienté **corpus francophone** — les données, le tokenizer et les filtres de qualité sont configurés pour le français.
> Cependant, on peut facilement adapter le pipeline à d'autres langues en modifiant les sources de données et les paramètres du tokenizer.

---

## 🏗️ Architecture

```
Tokens [batch, seq_length]
         │
         ▼
┌─────────────────────────────┐
│    Embedding Layer          │  Token Embedding + Position Embedding
│    (+ LayerNorm + Dropout)  │  Sparse gradients, weight tying-ready
└─────────────┬───────────────┘
              │  [batch, seq, embedding_dim]
              ▼
┌─────────────────────────────┐
│   Transformer Block × N     │  N = 4 (configurable)
│                             │
│  ┌───────────────────────┐  │
│  │  Pre-LayerNorm        │  │
│  │  Multi-Head Attention │  │  Flash Attention (sdpa), KV Cache
│  │  + Residual           │  │  Causal mask (triangulaire inférieur)
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Pre-LayerNorm        │  │
│  │  MLP (FFN × 4)        │  │  GELU, Dropout, scaled init GPT-2
│  │  + Residual           │  │
│  └───────────────────────┘  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Final LayerNorm            │
│  Output Projection          │  Weight tying avec token_emb
│  -> [batch, seq, vocab_size] │
└─────────────────────────────┘
```

### Paramètres de référence (config `small`)

| Paramètre        | Valeur     |
|------------------|------------|
| `vocab_size`     | 30 000     |
| `embedding_dim`  | 384        |
| `num_heads`      | 8          |
| `num_blocks`     | 4          |
| `seq_length`     | 256        |
| `dropout`        | 0.1        |
| `batch_size`     | 8 × 4 grad accum = **32** |

---

## 🔄 Pipeline complet

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. CORPUS          WebCrawler (async BFS) + Ingest local        │
│     ─────────────►  Wikipedia ZIM / PDF / TXT / HTML             │
│                     SQLite (crawler.db) + JSONL                  │
│                                                                  │
│  2. NETTOYAGE       build_clean_dataset.py                       │
│     ─────────────►  Chunking phrase-aware, LSH near-dedup        │
│                     dataset.jsonl propre                         │
│                                                                  │
│  3. TOKENISATION    BPE from scratch (optimisé heap)             │
│     ─────────────►  Vocab 30k tokens, encode_corpus()            │
│                     bpe_model.json                               │
│                                                                  │
│  4. DATASET         TextDataset — séquences glissantes           │
│     ─────────────►  input[t:t+seq] -> target[t+1:t+seq+1]        │
│                                                                  │
│  5. ENTRAÎNEMENT    AdamW + cosine decay + warmup                │
│     ─────────────►  AMP (float16), grad accumulation             │
│                     Checkpoints async, early resume              │
│                                                                  │
│  6. INFÉRENCE       KV Cache (O(1) par token)                    │
│     ─────────────►  Top-k sampling, température                  │
│                     Mode interactif reprompt                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Structure du projet

```
Portr-AI/
│
├── main.py                     # Point d'entrée principal du pipeline
│
├── data/
│   ├── dataset.py              # TextDataset (séquences glissantes)
│   ├── DataBase/
│   │   └── database.py         # SQLite + FTS5, déduplication SHA-256
│   └── datasets/
│       ├── crawler.db          # Base brute du crawler
│       └── dataset.jsonl       # Corpus nettoyé prêt à l'entraînement
│
├── tokenizer/
│   ├── tokenizer.py            # BPE from scratch (heap + lazy deletion)
│   ├── TokenizeData.py         # Wrapper streaming d'entraînement BPE
│   └── token/
│       └── bpe_model.json      # Modèle BPE sauvegardé
│
├── model/
│   ├── embedding.py            # Token + Position Embedding (sparse grads)
│   ├── attention.py            # Multi-Head Attention + Flash Attention + KV Cache
│   ├── transformer_block.py    # TransformerBlock (pre-norm, MLP GELU)
│   └── gpt_model.py            # Modèle GPT complet + weight tying + scaled init
│
├── training/
│   ├── train.py                # Boucle d'entraînement (AMP, grad accum, async ckpt)
│   ├── loss.py                 # Cross-entropy + label smoothing + z-loss
│   └── optimizer.py            # AdamW (fused/foreach, decay/no-decay groups)
│
├── inference/
│   └── generate.py             # Génération avec KV Cache, décodage BPE
│
├── Crawler/
│   ├── webcrawler.py           # Crawler async BFS (aiohttp, trafilatura)
│   └── config.py               # Configuration du crawler
│
├── artifacts/
│   ├── checkpoints/            # Checkpoints d'entraînement (.pt)
│   ├── models/                 # Modèle final pour inférence
│   └── cache/                  # Cache binaire des tokens encodés
│
├── crawl.py                    # CLI du crawler web
├── ingest_local.py             # Ingestion fichiers locaux (PDF, ZIM, TXT)
├── build_clean_dataset.py      # Reconstruction dataset propre depuis la DB
└── clean_db.py                 # Audit et nettoyage de la base SQLite
```

---

## ⚙️ Installation

### Prérequis

- Python **3.10+**
- CUDA (optionnel, recommandé pour l'entraînement)
- ~4 Go de RAM minimum, 8+ Go recommandés

### 1. Cloner le dépôt

```bash
git clone https://github.com/Por-tra/Portr-AI.git
cd Portr-AI
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
# Installer d'abord les dépendances listées
pip install -r requirements.txt

# Remarque : pour PyTorch, utilisez l'installateur officiel en fonction
# de votre CUDA / CPU. Exemple (CUDA 12.1) :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optionnel (PDF + ZIM) :
pip install pdfminer.six libzim
```

---

## 🚀 Utilisation

### 1. Collecte de données

#### Crawler web (async BFS)

```bash
# Crawler avec les URLs définies dans Crawler/config.py
python crawl.py

# Passer des URLs et options en argument
python crawl.py --url https://example.com --max-pages 5000 --max-depth 5
```

### Configuration

- Éditez `Crawler/config.py` pour définir : `start_urls`, `stay_on_domain`,
  `max_pages`, `num_workers`, `user_agents`, `proxies`, `db_file` et `jsonl_output`.
- Les chemins `db_file` et `jsonl_output` sont relatifs à la racine du dépôt ;
  assurez-vous que `data/datasets/` existe et est accessible en écriture.

Exemple : pour démarrer sur un site francophone, remplacez `start_urls` par
`["https://example-fr-site.org"]` et ajustez `max_pages`/`max_depth`.

#### Ingestion de fichiers locaux

```bash
# Sélecteur de fichiers GUI
python ingest_local.py

# Fichiers spécifiques (PDF, TXT, Markdown)
python ingest_local.py --files document.pdf notes.txt

# Wikipedia FR complet (fichier ZIM Kiwix — peut prendre plusieurs heures)
python ingest_local.py --files wikipedia_fr.zim \
    --commit-every 200 \
    --flush-every 200 \
    --progress-every 250 \
    --checkpoint-every 2000

# Dossier entier (récursif)
python ingest_local.py --dir ./mes_documents -r
```

### 2. Nettoyage et prétraitement

#### Audit et nettoyage de la base

```bash
# Voir les statistiques
python clean_db.py --stats

# Tout nettoyer (doublons, pages courtes, contenu vide) + export
python clean_db.py --all

# Supprimer les doublons uniquement
python clean_db.py --remove-dupes

# Supprimer les pages avec moins de 100 mots
python clean_db.py --remove-short 100
```

#### Reconstruire un dataset propre

```bash
# Avec les options par défaut (chunks de 700 mots, sans overlap)
python build_clean_dataset.py

# Personnalisé
python build_clean_dataset.py \
    --db data/datasets/crawler.db \
    --jsonl data/datasets/dataset.jsonl \
    --chunk-size 700 \
    --chunk-overlap 0 \
    --min-chars 180 \
    --near-dup-threshold 0.8
```

> **Note** : le chunking est **phrase-aware** — aucune coupure au milieu d'une phrase. La déduplication utilise LSH banding O(N) pour les near-duplicates.

### 3. Entraînement

```bash
python main.py
```

Le pipeline interactif propose de :
- choisir le mode du tokenizer (`auto` / `use_existing` / `retrain`)
- choisir le mode du cache d'encodage (`auto` / `use_precomputed` / `recompute`)

L'entraînement reprend automatiquement depuis le dernier checkpoint si `RESUME_FROM_CHECKPOINT = True`.

#### Suivi de l'entraînement

```
[train] epoch=1/3 step=20 loss=6.8421 lr=1.00e-05 grad_norm=0.842
[train] epoch=1/3 step=40 loss=5.2134 lr=2.00e-05 grad_norm=0.631
...
[train] epoch 1 done | avg_loss=4.3201 | steps=2048 | time=312.4s
[train] final loss: 3.2145 | perplexity: 24.87
```

### 4. Inférence

L'inférence se lance automatiquement à la fin de `main.py`, ou directement depuis l'artefact final :

```python
# Dans main.py : définir
INFERENCE_ONLY_FROM_FINAL_MODEL = True
INFER_PROMPT = "Bonjour, je suis un modèle de langage"
INFER_MAX_NEW_TOKENS = 80
INFER_TEMPERATURE = 0.9
INFER_TOP_K = 20
```

En fin de run, le **mode reprompt interactif** permet de tester plusieurs prompts sans relancer le pipeline :

```
[main] mode reprompt actif
prompt> Raconte-moi une histoire
[reprompt] tokens: 52
[reprompt] text: Raconte-moi une histoire sur un vieux château...

prompt> :quit
```

---

## 🎛️ Hyperparamètres

Les paramètres clés sont centralisés dans `main.py` :

| Paramètre              | Valeur par défaut | Description                          |
|------------------------|-------------------|--------------------------------------|
| `VOCAB_SIZE`           | `30 000`          | Taille du vocabulaire BPE            |
| `EMBEDDING_DIM`        | `384`             | Dimension des embeddings             |
| `NUM_HEADS`            | `8`               | Têtes d'attention                    |
| `NUM_BLOCKS`           | `4`               | Blocs Transformer                    |
| `SEQ_LENGTH`           | `256`             | Longueur de séquence                 |
| `BATCH_SIZE`           | `8`               | Taille de batch par GPU step         |
| `GRAD_ACCUM_STEPS`     | `4`               | Batch effectif = 8 × 4 = **32**      |
| `LEARNING_RATE`        | `1e-4`            | Learning rate initial                |
| `WEIGHT_DECAY`         | `0.01`            | Pénalité L2 (AdamW)                  |
| `EPOCHS`               | `3`               | Nombre d'epochs                      |
| `WARMUP_STEPS`         | `2% du total`     | Steps de warmup linéaire             |
| `LABEL_SMOOTHING`      | `0.1`             | Réduit l'overconfidence              |
| `Z_LOSS_WEIGHT`        | `1e-4`            | Stabilise les logits (z-loss)        |
| `USE_AMP`              | `True`            | Mixed precision (float16 sur GPU)    |
| `CHECKPOINT_EVERY`     | `500`             | Sauvegarde checkpoints (steps)       |

---

## Optimisations techniques

### Tokenizer BPE

| Problème original | Solution |
|---|---|
| `count_pairs()` O(N × V) | Comptage **incrémental** — seules les paires voisines du token fusionné sont mises à jour |
| `most_frequent_pair()` O(P) | **Heap max** avec lazy-deletion -> O(log P) |
| `encode_corpus()` lent | **Cache word->ids** + `array('i')` compact en mémoire |

### Attention

- **Flash Attention** via `F.scaled_dot_product_attention` : évite de matérialiser la matrice [B, H, T, T] — 2-4× moins de VRAM
- **KV Cache** : pendant le decode, seul le token courant est projeté (T=1), les K/V des étapes précédentes viennent du cache -> O(1) par token
- Correction `is_causal` : `False` en mode decode (sinon le cache est ignoré)

### Training

- **AMP float16** sur GPU, `GradScaler` pour la stabilité
- **Gradient accumulation** : batch effectif sans saturer la VRAM
- **Checkpoints asynchrones** : écriture disque dans un thread daemon (non-bloquant)
- **`loss.item()` différé** : pas de synchronisation GPU->CPU à chaque step, uniquement au `log_interval`
- **AdamW fused** kernel (PyTorch ≥ 2.0) + separation decay/no-decay groups

### Dataset & Corpus

- **LSH banding O(N)** pour la déduplication near-duplicate (vs O(N²) naïf)
- **Chunking phrase-aware** : aucune coupure au milieu d'une phrase
- **Encodage streamé** : le corpus JSONL n'est jamais chargé entièrement en RAM
- **Cache binaire** mémoire-mappé (`uint32`, 4 octets/token)

---

## Roadmap

- [x] **Phase 1** — Tokenizer BPE from scratch (optimisé)
- [x] **Phase 2** — Web Crawler async + Ingest local (ZIM/PDF/TXT)
- [x] **Phase 3** — Dataset builder (chunking phrase-aware, LSH near-dedup)
- [x] **Phase 4** — Embedding (token + position, sparse grads)
- [x] **Phase 5** — Self-Attention + Flash Attention + KV Cache
- [x] **Phase 6** — Transformer Block (pre-norm, MLP GELU)
- [x] **Phase 7** — Modèle GPT complet (weight tying, scaled init)
- [x] **Phase 8** — Training loop (AMP, grad accum, async checkpoints)
- [x] **Phase 9** — Génération de texte (Top-k, température, mode reprompt)
- [ ] **Phase 10** — Évaluation (perplexité, BLEU, human eval)
- [ ] **Phase 11** — Scaling (plus de données, modèle medium)
- [ ] **Phase 12** — Flash Attention v2, Gradient Checkpointing, KV Cache étendu

---

## Ressources

### Articles fondateurs

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) — GPT-2 Paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar

### Tokenisation BPE

- [BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — Sebastian Raschka
- [Video : BPE expliqué](https://youtu.be/VrZTHO1E76w) — YouTube

### Documentation du projet

> **📖 Documentation complète, tutoriels et guides d'utilisation :**
>
> [![Documentation Wiki](https://img.shields.io/badge/📖_Wiki_Portr--AI-por--tra.github.io-4A90D9?style=for-the-badge&logo=github)](https://por-tra.github.io/Portr-AI-DOC/)

---

## Auteur

**Lucas** — Projet personnel d'apprentissage et d'implémentation from scratch d'un LLM en français.

---

<div align="center">


[![Documentation](https://img.shields.io/badge/Documentation-por--tra.github.io%2FPortr--AI--DOC-blue?style=flat-square)](https://por-tra.github.io/Portr-AI-DOC/)

</div>