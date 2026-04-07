<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Language-French_corpus-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Architecture-GPT_Transformer-7952B3?style=for-the-badge"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

```
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝
```

### **A French GPT-style LLM, built entirely from scratch**
*Understanding every component — from raw corpus to text generation*

<br/>

[![📖 Documentation](https://img.shields.io/badge/📖_Documentation-por--tra.github.io-4A90D9?style=for-the-badge)](https://por-tra.github.io/Portr-AI-DOC/)

> 🇫🇷 [Version française disponible plus bas](#-version-française)

</div>

---

> **Note:** The project has not been fully tested on the complete dataset due to hardware constraints (GPU VRAM, training time). The pipeline is complete and functional — compute is the only bottleneck.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Full Pipeline](#-full-pipeline)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [1. Data collection](#1-data-collection)
  - [2. Cleaning and preprocessing](#2-cleaning-and-preprocessing)
  - [3. Training](#3-training)
  - [4. Inference](#4-inference)
- [Hyperparameters](#-hyperparameters)
- [Technical optimizations](#-technical-optimizations)
- [Roadmap](#-roadmap)
- [Resources](#-resources)
- [Version française](#-version-française)

---

## 🧠 Overview

**Portr-AI** is an autoregressive language model (GPT-style), built entirely **from scratch in Python/PyTorch** — no pretrained model, no library shortcut.

The goal is educational and technical: **master every brick** of the Transformer architecture, from BPE tokenization to text generation, including GPU training.

> 💡 The model targets a **French corpus** — data, tokenizer and quality filters are configured for French. The pipeline can be adapted to other languages by changing data sources and tokenizer settings.

---

## 🏗️ Architecture

```
Tokens [batch, seq_length]
         │
         ▼
┌─────────────────────────────┐
│    Embedding Layer          │  Token Embedding + Position Embedding
│    (+ Dropout)              │  Sparse gradients, weight tying-ready
└─────────────┬───────────────┘
              │  [batch, seq, embedding_dim]
              ▼
┌─────────────────────────────┐
│   Transformer Block × N     │  N = 4 (configurable)
│                             │
│  ┌───────────────────────┐  │
│  │  Pre-LayerNorm        │  │
│  │  Multi-Head Attention │  │  Flash Attention (sdpa) + KV Cache
│  │  + Residual           │  │  Causal mask, is_causal fix for decode
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  Pre-LayerNorm        │  │
│  │  MLP (FFN × 4)        │  │  GELU, Dropout, GPT-2 scaled init
│  │  + Residual           │  │
│  └───────────────────────┘  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Final LayerNorm            │
│  Output Projection          │  Weight tying with token_emb
│  → [batch, seq, vocab_size] │
└─────────────────────────────┘
```

### Reference parameters (`small` config)

| Parameter        | Value      |
|------------------|------------|
| `vocab_size`     | 30 000     |
| `embedding_dim`  | 384        |
| `num_heads`      | 8          |
| `num_blocks`     | 4          |
| `seq_length`     | 256        |
| `dropout`        | 0.1        |
| `batch_size`     | 8 × 4 grad accum = **32 effective** |
| Total params     | ~18.7M     |

---

## 🔄 Full Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  1. CORPUS          Async BFS WebCrawler + local ingest          │
│     ─────────────►  Wikipedia ZIM / PDF / TXT / HTML             │
│                     SQLite (crawler.db) + JSONL                  │
│                                                                  │
│  2. CLEANING        build_clean_dataset.py                       │
│     ─────────────►  Sentence-aware chunking, near-dedup          │
│                     Clean dataset.jsonl                          │
│                                                                  │
│  3. TOKENIZATION    BPE from scratch (heap-optimized)            │
│     ─────────────►  30k vocab, encode_corpus()                   │
│                     bpe_model.json                               │
│                                                                  │
│  4. DATASET         TextDataset — sliding windows                │
│     ─────────────►  input[t:t+seq] → target[t+1:t+seq+1]        │
│                                                                  │
│  5. TRAINING        AdamW + cosine decay + warmup                │
│     ─────────────►  AMP (float16), grad accumulation             │
│                     Async checkpoints, auto-resume               │
│                                                                  │
│  6. INFERENCE       KV Cache (O(1) per token)                    │
│     ─────────────►  Top-k sampling, temperature                  │
│                     Interactive reprompt mode                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Portr-AI/
│
├── main.py                     # Main pipeline entry point
│
├── data/
│   ├── dataset.py              # TextDataset (sliding windows)
│   ├── DataBase/
│   │   └── database.py         # SQLite + FTS5, SHA-256 deduplication
│   └── datasets/
│       ├── crawler.db          # Raw crawler database
│       └── dataset.jsonl       # Clean corpus ready for training
│
├── tokenizer/
│   ├── tokenizer.py            # BPE from scratch (heap + lazy deletion)
│   ├── TokenizeData.py         # Streaming BPE training wrapper
│   └── token/
│       └── bpe_model.json      # Saved BPE model
│
├── model/
│   ├── embedding.py            # Token + Position Embedding (sparse grads)
│   ├── attention.py            # Multi-Head Attention + Flash Attention + KV Cache
│   ├── transformer_block.py    # TransformerBlock (pre-norm, GELU MLP)
│   └── gpt_model.py            # Full GPT model + weight tying + scaled init
│
├── training/
│   ├── train.py                # Training loop (AMP, grad accum, async ckpt)
│   ├── loss.py                 # Cross-entropy + label smoothing + z-loss
│   └── optimizer.py            # AdamW (fused/foreach, decay/no-decay groups)
│
├── inference/
│   └── generate.py             # Generation with KV Cache, BPE decoding
│
├── Crawler/
│   ├── webcrawler.py           # Async BFS crawler (aiohttp, trafilatura)
│   └── config.py               # Crawler configuration
│
├── artifacts/
│   ├── checkpoints/            # Training checkpoints (.pt)
│   ├── models/                 # Final inference model
│   └── cache/                  # Binary token cache (memory-mapped)
│
├── crawl.py                    # Crawler CLI
├── ingest_local.py             # Local file ingestion (PDF, ZIM, TXT)
├── build_clean_dataset.py      # Clean dataset builder from SQLite
└── clean_db.py                 # SQLite audit and cleanup
```

---

## ⚙️ Installation

### Requirements

- Python **3.10+**
- CUDA (optional, strongly recommended for training)
- ~4 GB RAM minimum, 8+ GB recommended

### 1. Clone the repository

```bash
git clone https://github.com/Por-tra/Portr-AI.git
cd Portr-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt

# For PyTorch, use the official installer matching your CUDA version.
# Example (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional (PDF + ZIM support):
pip install pdfminer.six libzim
```

---

## 🚀 Usage

### 1. Data Collection

#### Web crawler (async BFS)

```bash
# Crawl using URLs defined in Crawler/config.py
python crawl.py

# Pass URLs and options directly
python crawl.py --url https://example.com --max-pages 5000 --max-depth 5
```

**Configuration:** edit `Crawler/config.py` to set `start_urls`, `stay_on_domain`, `max_pages`, `num_workers`, `db_file`, and `jsonl_output`.

#### Local file ingestion

```bash
# GUI file selector
python ingest_local.py

# Specific files (PDF, TXT, Markdown)
python ingest_local.py --files document.pdf notes.txt

# Full French Wikipedia (ZIM file from Kiwix — may take several hours)
python ingest_local.py --files wikipedia_fr.zim \
    --commit-every 200 --progress-every 250

# Recursive folder
python ingest_local.py --dir ./my_documents -r
```

### 2. Cleaning and Preprocessing

#### Database audit and cleanup

```bash
python clean_db.py --stats                  # Show statistics
python clean_db.py --all                    # Full cleanup + export
python clean_db.py --remove-dupes           # Remove duplicates only
python clean_db.py --remove-short 100       # Remove pages under 100 words
```

#### Rebuild a clean dataset

```bash
# Default options (700-word chunks, no overlap)
python build_clean_dataset.py

# Custom
python build_clean_dataset.py \
    --db data/datasets/crawler.db \
    --jsonl data/datasets/dataset.jsonl \
    --chunk-size 700 \
    --chunk-overlap 0 \
    --min-chars 180 \
    --near-dup-threshold 0.8
```

> **Note:** chunking is **sentence-aware** — no mid-sentence cuts. Near-duplicate detection uses Jaccard 8-gram fingerprinting.

### 3. Training

```bash
python main.py
```

The interactive pipeline will ask you to choose:
- tokenizer mode (`auto` / `use_existing` / `retrain`)
- encoding cache mode (`auto` / `use_precomputed` / `recompute`)

Training resumes automatically from the last checkpoint if `RESUME_FROM_CHECKPOINT = True`.

**Training output:**

```
[train] epoch=1/3 step=20  loss=10.35 lr=2.50e-07 grad_norm=29.3
[train] epoch=1/3 step=100 loss=8.17  lr=1.25e-06 grad_norm=8.7
...
[train] epoch 1 done | avg_loss=4.32 | steps=54443584 | time=...
[train] final loss: 3.21 | perplexity: 24.87
```

**Visualization:** training curves (loss, perplexity, learning rate) are saved in `artifacts/` and can be visualized with TensorBoard:

![Aperçu 1](asset/courbe.png)
![Aperçu 2](asset/courbe2.png)

### 4. Inference

Inference runs automatically at the end of `main.py`, or directly from the saved artifact:

```python
# In main.py:
INFERENCE_ONLY_FROM_FINAL_MODEL = True
INFER_PROMPT = "Hello, I am a language model"
INFER_MAX_NEW_TOKENS = 80
INFER_TEMPERATURE = 0.9
INFER_TOP_K = 20
```

At the end of the run, an **interactive reprompt mode** lets you test multiple prompts without restarting the pipeline:

```
[main] reprompt mode active
prompt> Tell me a story
[reprompt] tokens: 52
[reprompt] text: Tell me a story about an old castle...

prompt> :quit
```

---

## 🎛️ Hyperparameters

All key parameters are centralized in `main.py`. `WARMUP_STEPS` is computed automatically as 2% of total steps at runtime.

| Parameter              | Default           | Description                          |
|------------------------|-------------------|--------------------------------------|
| `VOCAB_SIZE`           | `30 000`          | BPE vocabulary size                  |
| `EMBEDDING_DIM`        | `384`             | Embedding dimension                  |
| `NUM_HEADS`            | `8`               | Attention heads                      |
| `NUM_BLOCKS`           | `4`               | Transformer blocks                   |
| `SEQ_LENGTH`           | `256`             | Sequence length                      |
| `BATCH_SIZE`           | `8`               | Batch size per step                  |
| `GRAD_ACCUM_STEPS`     | `4`               | Effective batch = 8 × 4 = **32**     |
| `LEARNING_RATE`        | `1e-4`            | Peak learning rate                   |
| `WEIGHT_DECAY`         | `0.01`            | L2 penalty (AdamW)                   |
| `EPOCHS`               | `3`               | Number of epochs                     |
| `WARMUP_STEPS`         | `~2% of total`    | Auto-computed linear warmup          |
| `LABEL_SMOOTHING`      | `0.1`             | Reduces overconfidence               |
| `Z_LOSS_WEIGHT`        | `1e-4`            | Logit stabilization (z-loss)         |
| `USE_AMP`              | `True`            | Mixed precision (float16 on GPU)     |
| `CHECKPOINT_EVERY`     | `500`             | Async checkpoint frequency (steps)   |

---

## ⚡ Technical Optimizations

### BPE Tokenizer

| Problem | Solution |
|---|---|
| `count_pairs()` O(N × V) naive | **Incremental counting** — only neighboring pairs of the merged token are updated |
| `most_frequent_pair()` O(P) | **Max-heap** with lazy deletion → O(log P) |
| Slow `encode_corpus()` | **Word→ids cache** + compact `array('i')` buffer |

### Attention

- **Flash Attention** via `F.scaled_dot_product_attention`: avoids materializing the [B, H, T, T] score matrix — 2-4× less VRAM
- **KV Cache**: during decode, only the current token is projected (T=1), past K/V come from cache → O(1) per token
- **`is_causal` fix**: set to `False` during decode (when `past_kv is not None`) — otherwise the entire KV cache is masked out and ignored

### Training Loop

- **AMP float16** on GPU with `GradScaler` for numerical stability
- **Gradient accumulation**: large effective batch without saturating VRAM
- **Async checkpoints**: disk writes happen in a daemon thread — training is never blocked by I/O
- **Deferred `loss.item()`**: no GPU→CPU sync at every step, only at `log_interval`
- **Fused AdamW** kernel (PyTorch ≥ 2.0) + separate decay/no-decay parameter groups
- **Cosine LR decay** with linear warmup auto-computed from dataset size

### Dataset & Corpus

- **Sentence-aware chunking**: no sentence is ever cut mid-way
- **Jaccard 8-gram near-dedup**: catches paraphrases and overlapping chunks without external dependencies
- **Streaming encoding**: the JSONL corpus is never fully loaded into RAM
- **Memory-mapped binary cache**: `uint32`, 4 bytes/token, loaded via `torch.from_file`

---

## 🗺️ Roadmap

- [x] **Phase 1** — BPE tokenizer from scratch (heap-optimized)
- [x] **Phase 2** — Async web crawler + local ingest (ZIM/PDF/TXT)
- [x] **Phase 3** — Dataset builder (sentence-aware chunking, near-dedup)
- [x] **Phase 4** — Embedding layer (token + position, sparse gradients)
- [x] **Phase 5** — Self-Attention + Flash Attention + KV Cache
- [x] **Phase 6** — Transformer Block (pre-norm, GELU MLP)
- [x] **Phase 7** — Full GPT model (weight tying, GPT-2 scaled init)
- [x] **Phase 8** — Training loop (AMP, grad accum, async checkpoints)
- [x] **Phase 9** — Text generation (top-k, temperature, reprompt mode)
- [ ] **Phase 10** — Evaluation (perplexity curves, BLEU, human eval)
- [ ] **Phase 11** — Scaling (larger dataset, medium model ~50M params)
- [ ] **Phase 12** — Flash Attention v2, gradient checkpointing, extended KV cache

---

## 📚 Resources

### Foundational papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://openai.com/research/better-language-models) — GPT-2 paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar

### BPE tokenization

- [BPE from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html) — Sebastian Raschka
- [BPE explained (video)](https://youtu.be/VrZTHO1E76w)

### Full project documentation

[![Documentation](https://img.shields.io/badge/📖_Wiki_Portr--AI-por--tra.github.io-4A90D9?style=for-the-badge&logo=github)](https://por-tra.github.io/Portr-AI-DOC/)

---

---

## 🇫🇷 Version française

<div align="center">

### **Un LLM GPT français, implémenté from scratch**
*Comprendre chaque rouage — du corpus brut à la génération de texte*

</div>

> 🇬🇧 [English version above](#-overview)

---

## Vue d'ensemble

**Portr-AI** est un modèle de langage auto-régressif de type GPT, construit entièrement **from scratch en Python/PyTorch**, sans utilisation d'aucun modèle pré-entraîné.

L'objectif est pédagogique et technique : **maîtriser chaque brique** de l'architecture Transformer, de la tokenisation BPE jusqu'à la génération de texte, en passant par l'entraînement sur GPU.

> 💡 Le modèle est orienté **corpus francophone**. Le pipeline peut être adapté à d'autres langues en modifiant les sources de données et les paramètres du tokenizer.

> **Note :** Le projet n'a pas pu être testé à 100% sur le dataset complet à cause de contraintes matérielles (VRAM GPU, temps d'entraînement). Le pipeline est complet et fonctionnel — le compute est le seul facteur limitant.

---

## Installation

### Prérequis

- Python **3.10+**
- CUDA (optionnel, fortement recommandé pour l'entraînement)
- ~4 Go de RAM minimum, 8+ Go recommandés

```bash
git clone https://github.com/Por-tra/Portr-AI.git
cd Portr-AI
python -m venv .venv
.venv\Scripts\activate           # Windows
# ou
source .venv/bin/activate        # Linux/macOS

pip install -r requirements.txt
```

---

## Utilisation rapide

### Collecte de données

```bash
# Crawler web
python crawl.py --url https://example.com --max-pages 5000

# Fichiers locaux (PDF, TXT, ZIM Wikipedia)
python ingest_local.py --files document.pdf
python ingest_local.py --dir ./mes_documents -r
```

### Nettoyage du dataset

```bash
python clean_db.py --all
python build_clean_dataset.py --chunk-size 700 --chunk-overlap 0
```

### Entraînement

```bash
python main.py
```

Le pipeline interactif propose de choisir le mode tokenizer et le mode cache. L'entraînement reprend automatiquement depuis le dernier checkpoint.

### Inférence

```python
# Dans main.py :
INFERENCE_ONLY_FROM_FINAL_MODEL = True
INFER_PROMPT = "Bonjour, je suis un modèle de langage"
INFER_MAX_NEW_TOKENS = 80
```

---

## Optimisations clés

| Composant | Optimisation |
|---|---|
| Tokenizer BPE | Comptage incrémental + heap max + cache word→ids |
| Attention | Flash Attention (sdpa) + KV Cache O(1) + fix `is_causal` en decode |
| Training | AMP float16 + grad accum + checkpoints async + `loss.item()` différé |
| Dataset | Chunking phrase-aware + near-dedup Jaccard + cache binaire mémoire-mappé |

---

## Auteur

**Lucas** — Projet personnel d'apprentissage et d'implémentation from scratch d'un LLM en français.

[![Documentation](https://img.shields.io/badge/📖_Documentation-por--tra.github.io-4A90D9?style=flat-square)](https://por-tra.github.io/Portr-AI-DOC/)