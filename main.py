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

import json
from pathlib import Path
import sys
from collections import Counter
from functools import partial

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
td.VOCAB_SIZE = 30_000  # 8k-16k minimum recommande pour du francais
td.LIMIT_CORPUS = td.VOCAB_SIZE * 10

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
EPOCHS = 43
MAX_STEPS_PER_EPOCH = 1_000  # evite un run infini sur gros dataset
GRAD_CLIP = 1.0
LOG_INTERVAL = 20
WARMUP_STEPS = 1049
LR_DECAY_FACTOR = 0.1
USE_AMP = True
GRAD_ACCUM_STEPS = 1

# Loss config
LOSS_IGNORE_INDEX = -100
LABEL_SMOOTHING = 0.0
Z_LOSS_WEIGHT = 0.0

# Pipeline controls
RUN_TOKENIZER_TRAINING = True
# Tokenizer mode:
# - "auto": charge modèle existant sinon entraine
# - "use_existing": utilise uniquement le modèle deja entrainé
# - "retrain": force le ré-entraînement
TOKENIZER_MODE = "auto"
DATASET_JSONL_PATH = "data/datasets/dataset.jsonl"
CHECKPOINT_DIR = "artifacts/checkpoints"
CHECKPOINT_NAME = "gpt_latest.pt"
TOKENIZER_MODEL_PATH = "tokenizer/token/bpe_model.json"

# Cache controls for heavy preprocessing
# Modes:
# - "auto": use cache when valid, otherwise recompute and refresh cache
# - "use_precomputed": use cache only (fail if missing/invalid)
# - "recompute": always recompute and overwrite cache
ENCODING_CACHE_MODE = "auto"
ENCODING_CACHE_PATH = "artifacts/cache/encoded_tokens.json"

# Inference config
INFER_PROMPT = "bonjour je suis"
INFER_MAX_NEW_TOKENS = 40
INFER_TEMPERATURE = 0.9
INFER_TOP_K = 20
ENABLE_INTERACTIVE_REPROMPT = True


def _select_cache_mode_console() -> None:
    """
    Let the user choose cache strategy at runtime from console input.

    - use_precomputed: force cached encoded data
    - recompute: force heavy encoding recomputation
    - auto: use cache if valid, otherwise recompute
    """
    global ENCODING_CACHE_MODE

    print("\n[main] Choix des donnees d'encodage:")
    print("  1) Utiliser les donnees deja chargees (cache pre-calcule)")
    print("  2) Refaire les calculs d'encodage (recompute)")
    print("  3) Auto (cache si valide sinon recompute)")
    print(f"[main] mode actuel: {ENCODING_CACHE_MODE}")

    try:
        choice = input("[main] Votre choix [1/2/3] (Entree = garder actuel): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[main] aucun choix saisi, mode conserve")
        return

    if not choice:
        print(f"[main] mode conserve: {ENCODING_CACHE_MODE}")
        return

    mapping = {
        "1": "use_precomputed",
        "2": "recompute",
        "3": "auto",
        "use_precomputed": "use_precomputed",
        "recompute": "recompute",
        "auto": "auto",
    }

    resolved = mapping.get(choice.lower())
    if resolved is None:
        print(f"[main] choix invalide '{choice}', mode conserve: {ENCODING_CACHE_MODE}")
        return

    ENCODING_CACHE_MODE = resolved
    print(f"[main] mode selectionne: {ENCODING_CACHE_MODE}")


def _validate_files() -> bool:
    if not Path(DATASET_JSONL_PATH).exists():
        print(f"[main] dataset introuvable: {DATASET_JSONL_PATH}")
        return False
    if ENCODING_CACHE_MODE not in {"auto", "use_precomputed", "recompute"}:
        print(f"[main] ENCODING_CACHE_MODE invalide: {ENCODING_CACHE_MODE}")
        return False
    if TOKENIZER_MODE not in {"auto", "use_existing", "retrain"}:
        print(f"[main] TOKENIZER_MODE invalide: {TOKENIZER_MODE}")
        return False
    return True


def _select_tokenizer_mode_console() -> None:
    """Console choice for tokenizer behavior (independent from encoding cache)."""
    global TOKENIZER_MODE

    print("\n[main] Choix du tokenizer:")
    print("  1) Utiliser le tokenizer existant (pas de ré-entraînement)")
    print("  2) Re-entraîner le tokenizer (retrain)")
    print("  3) Auto (charger si existe, sinon entraîner)")
    print(f"[main] mode tokenizer actuel: {TOKENIZER_MODE}")

    try:
        choice = input("[main] Votre choix tokenizer [1/2/3] (Entree = garder actuel): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[main] aucun choix tokenizer saisi, mode conserve")
        return

    if not choice:
        print(f"[main] mode tokenizer conserve: {TOKENIZER_MODE}")
        return

    mapping = {
        "1": "use_existing",
        "2": "retrain",
        "3": "auto",
        "use_existing": "use_existing",
        "retrain": "retrain",
        "auto": "auto",
    }
    resolved = mapping.get(choice.lower())
    if resolved is None:
        print(f"[main] choix tokenizer invalide '{choice}', mode conserve: {TOKENIZER_MODE}")
        return

    TOKENIZER_MODE = resolved
    print(f"[main] mode tokenizer selectionne: {TOKENIZER_MODE}")


def _file_signature(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": path, "exists": False}
    st = p.stat()
    return {
        "path": path,
        "exists": True,
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _build_encoding_meta(tokenizer_model) -> dict:
    return {
        "dataset": _file_signature(DATASET_JSONL_PATH),
        "tokenizer_model": _file_signature(TOKENIZER_MODEL_PATH),
        "target_vocab_size": int(td.VOCAB_SIZE),
        "corpus_limit": int(td.LIMIT_CORPUS),
        "tokenizer_vocab_size": int(len(tokenizer_model.get_token_to_id())),
        "seq_length": int(SEQ_LENGTH),
    }


def _save_token_cache(tokens: list[int], meta: dict) -> None:
    cache_path = Path(ENCODING_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": meta,
        "token_count": len(tokens),
        "tokens": tokens,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[main] cache ecrit: {cache_path} ({len(tokens)} tokens)")


def _load_token_cache(expected_meta: dict) -> list[int] | None:
    cache_path = Path(ENCODING_CACHE_PATH)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    meta = payload.get("meta")
    tokens = payload.get("tokens")
    if not isinstance(meta, dict) or not isinstance(tokens, list):
        return None

    if meta != expected_meta:
        return None

    # Defensive conversion for JSON numbers
    try:
        return [int(x) for x in tokens]
    except (TypeError, ValueError):
        return None


def _get_or_encode_tokens(tokenizer_model) -> list[int]:
    expected_meta = _build_encoding_meta(tokenizer_model)

    if ENCODING_CACHE_MODE in {"auto", "use_precomputed"}:
        cached_tokens = _load_token_cache(expected_meta)
        if cached_tokens is not None:
            print("[main] cache valide trouve, encodage saute")
            return cached_tokens

        if ENCODING_CACHE_MODE == "use_precomputed":
            raise RuntimeError(
                "Mode use_precomputed actif, mais aucun cache valide n'a ete trouve. "
                "Passe ENCODING_CACHE_MODE='auto' ou 'recompute'."
            )

    print("[main] encodage du corpus (calcul lourd)...")
    tokens = tokenizer_model.encode_corpus(DATASET_JSONL_PATH, limit=td.LIMIT_CORPUS)
    _save_token_cache(tokens, expected_meta)
    return tokens


def _check_dataset_quality(path: str, sample_lines: int = 5000) -> bool:
    """Check JSONL integrity and basic text quality indicators."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        print(f"[data-check] fichier introuvable: {path}")
        return False

    total = 0
    parsed = 0
    malformed_json = 0
    missing_text = 0
    empty_text = 0
    short_text = 0
    long_text = 0
    char_counter = Counter()

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if sample_lines and total > sample_lines:
                break

            line = line.strip()
            if not line:
                empty_text += 1
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed_json += 1
                continue

            parsed += 1
            text = item.get("text")
            if text is None:
                missing_text += 1
                continue
            if not isinstance(text, str):
                missing_text += 1
                continue

            t = text.strip()
            if not t:
                empty_text += 1
                continue

            n = len(t)
            if n < 10:
                short_text += 1
            if n > 5000:
                long_text += 1
            char_counter.update(t)

    print("[data-check] resume qualite (echantillon):")
    print(f"[data-check] lignes lues: {total}")
    print(f"[data-check] json valides: {parsed}")
    print(f"[data-check] json invalides: {malformed_json}")
    print(f"[data-check] champ text manquant/invalide: {missing_text}")
    print(f"[data-check] textes vides: {empty_text}")
    print(f"[data-check] textes tres courts (<10): {short_text}")
    print(f"[data-check] textes tres longs (>5000): {long_text}")

    if char_counter:
        top_chars = char_counter.most_common(8)
        print(f"[data-check] caracteres frequents: {top_chars}")

    usable = parsed - missing_text - empty_text
    if usable <= 0:
        print("[data-check] aucune donnee exploitable detectee")
        return False
    if malformed_json > max(5, total // 20):
        print("[data-check] trop de lignes JSON invalides")
        return False

    return True


def _build_dataloader(tokenizer_model) -> tuple[DataLoader, int]:
    tokens = _get_or_encode_tokens(tokenizer_model)
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
        log_interval=LOG_INTERVAL,
        max_steps_per_epoch=MAX_STEPS_PER_EPOCH,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        warmup_steps=WARMUP_STEPS,
        lr_decay_factor=LR_DECAY_FACTOR,
        use_amp=USE_AMP,
        grad_accum_steps=GRAD_ACCUM_STEPS,
    )

    loss_fn = partial(
        autoregressive_cross_entropy,
        ignore_index=LOSS_IGNORE_INDEX,
        label_smoothing=LABEL_SMOOTHING,
        z_loss_weight=Z_LOSS_WEIGHT,
    )

    history = train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
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


def _interactive_reprompt_loop(model: GPT, tokenizer_model) -> None:
    """Allow repeated prompt tests without rerunning full pipeline."""
    if not ENABLE_INTERACTIVE_REPROMPT:
        return

    print("\n[main] mode reprompt actif")
    print("[main] Tape un prompt puis Entrer.")
    print("[main] Commandes: ':quit' pour sortir, ':help' pour aide.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    while True:
        try:
            user_input = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] reprompt termine")
            break

        if not user_input:
            continue

        if user_input in {":quit", ":q", "exit", "quit"}:
            print("[main] sortie reprompt")
            break

        if user_input == ":help":
            print("[main] Utilise un texte normal pour tester le modele.")
            print("[main] ':quit' pour quitter.")
            continue

        token_ids, text = generate_from_prompt(
            model=model,
            tokenizer=tokenizer_model,
            prompt=user_input,
            max_new_tokens=INFER_MAX_NEW_TOKENS,
            temperature=INFER_TEMPERATURE,
            top_k=INFER_TOP_K,
            device=device,
        )

        print(f"[reprompt] tokens: {len(token_ids)}")
        print(f"[reprompt] text:\n{text}\n")


def main() -> None:
    print("[main] pipeline start")

    _select_tokenizer_mode_console()
    _select_cache_mode_console()

    if not _validate_files():
        return

    if not _check_dataset_quality(DATASET_JSONL_PATH):
        print("[main] dataset quality check echoue, pipeline stop")
        return

    # 1) Tokenization/training tokenizer model
    if RUN_TOKENIZER_TRAINING:
        print("[main] tokenization step...")
        td.TokenizeData(mode=TOKENIZER_MODE)

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

    # 7) Interactive reprompt tests
    _interactive_reprompt_loop(model, tokenizer_model)

    print("[main] pipeline complete")


if __name__ == "__main__":
    main()
