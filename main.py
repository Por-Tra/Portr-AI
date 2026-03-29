"""
main.py

Pipeline principal du projet:
1) Tokenisation
2) Encodage du corpus
3) Construction dataset/dataloader
4) Entrainement GPT
5) Sauvegarde checkpoint
6) Inference de test
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

import tokenizer.TokenizeData as td
from data.dataset import TextDataset
from training.loss import autoregressive_cross_entropy, perplexity_from_loss
from training.optimizer import create_adamw_optimizer
from training.train import TrainConfig, train_model
from inference.generate import generate_from_prompt


# Make model directory importable (gpt_model imports local files inside model/).
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.gpt_model import GPT


# -----------------------------------------------------------------------------
# Variable de configuration des donnees et du modele
# -----------------------------------------------------------------------------
td.VOCAB_SIZE = 50_000  # changer ca pour ajuster la taille du vocablaire

# Variable de configuration des sequences
SEQ_LENGTH = 128
BATCH_SIZE = 32

# Training config
EMBEDDING_DIM = 256
NUM_HEADS = 8
NUM_BLOCKS = 4
DROPOUT = 0.1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EPOCHS = 1
MAX_STEPS_PER_EPOCH = 200  # evite un run infini sur gros dataset
GRAD_CLIP = 1.0

# Pipeline controls
RUN_TOKENIZER_TRAINING = True
DATASET_JSONL_PATH = "data/datasets/dataset.jsonl"
CHECKPOINT_DIR = "artifacts/checkpoints"
CHECKPOINT_NAME = "gpt_latest.pt"

# Inference config
INFER_PROMPT = "bonjour je suis"
INFER_MAX_NEW_TOKENS = 40
INFER_TEMPERATURE = 0.9
INFER_TOP_K = 20


def _validate_files() -> bool:
    if not Path(DATASET_JSONL_PATH).exists():
        print(f"[main] dataset introuvable: {DATASET_JSONL_PATH}")
        return False
    return True


def _build_dataloader(tokenizer_model) -> tuple[DataLoader, int]:
    print("[main] encodage du corpus...")
    tokens = tokenizer_model.encode_corpus(DATASET_JSONL_PATH)
    print(f"[main] nb tokens encodes: {len(tokens)}")

    if len(tokens) <= SEQ_LENGTH:
        raise ValueError(
            f"Pas assez de tokens pour SEQ_LENGTH={SEQ_LENGTH}. "
            f"tokens={len(tokens)}"
        )

    dataset = TextDataset(tokens, SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"[main] dataset size: {len(dataset)}")
    batch_x, batch_y = next(iter(dataloader))
    print(f"[main] batch input shape: {tuple(batch_x.shape)}")
    print(f"[main] batch target shape: {tuple(batch_y.shape)}")
    return dataloader, len(tokens)


def _train_pipeline(dataloader: DataLoader, tokenizer_model) -> tuple[GPT, dict]:
    vocab_size = len(tokenizer_model.get_token_to_id())
    print(f"[main] vocab_size effective: {vocab_size}")

    model = GPT(
        vocab_size=vocab_size,
        seq_length=SEQ_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    )

    optimizer = create_adamw_optimizer(
        model=model,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_cfg = TrainConfig(
        epochs=EPOCHS,
        grad_clip=GRAD_CLIP,
        max_steps_per_epoch=MAX_STEPS_PER_EPOCH,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_name=CHECKPOINT_NAME,
    )

    history = train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=autoregressive_cross_entropy,
        config=train_cfg,
    )

    if history["epoch_losses"]:
        last_loss = history["epoch_losses"][-1]
        ppl = perplexity_from_loss(last_loss)
        print(f"[main] final loss: {last_loss:.4f} | perplexity: {ppl:.2f}")

    return model, history


def _run_inference(model: GPT, tokenizer_model) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    token_ids, text = generate_from_prompt(
        model=model,
        tokenizer=tokenizer_model,
        prompt=INFER_PROMPT,
        max_new_tokens=INFER_MAX_NEW_TOKENS,
        temperature=INFER_TEMPERATURE,
        top_k=INFER_TOP_K,
        device=device,
    )

    print("[main] inference done")
    print(f"[main] generated token count: {len(token_ids)}")
    print(f"[main] generated text:\n{text}")


def main() -> None:
    print("[main] pipeline start")

    if not _validate_files():
        return

    # 1) Tokenization/training tokenizer model
    if RUN_TOKENIZER_TRAINING:
        print("[main] tokenization step...")
        td.TokenizeData()

    # 2) Load tokenizer model
    print("[main] loading tokenizer...")
    try:
        tokenizer_model = td.Tokenizer.load()
    except FileNotFoundError:
        print("[main] tokenizer introuvable. Lance d'abord TokenizeData().")
        return

    # 3) Build dataloader from encoded corpus
    dataloader, _num_tokens = _build_dataloader(tokenizer_model)

    # 4) Train GPT
    model, history = _train_pipeline(dataloader, tokenizer_model)

    # 5) Checkpoint info
    checkpoint_path = history.get("checkpoint_path")
    print(f"[main] checkpoint path: {checkpoint_path}")

    # 6) Inference test
    _run_inference(model, tokenizer_model)

    print("[main] pipeline complete")


if __name__ == "__main__":
    main()
