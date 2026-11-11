# -*- coding: utf-8 -*-
"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : main.py

description du script:
    Pipeline principal du projet:
    1) Tokenisation
    2) Encodage du corpus
    3) Construction dataset/dataloader
    4) Entrainement GPT
    5) Sauvegarde checkpoint
    6) Inference de test
    7) Mode reprompt interactif (test de plusieurs prompts sans relancer le pipeline)
"""


from __future__ import annotations

import json
import os
import argparse
from pathlib import Path
import sys
from datetime import datetime, timezone
from uuid import uuid4
from array import array
from collections import Counter
from functools import partial

import torch
from torch.utils.data import DataLoader

from tokenizer.tokenizer import Tokenizer
from data.dataset import TextDataset, ResumableRandomSampler
from data.sft_dataset import SFTDataset
from training.loss import autoregressive_cross_entropy, perplexity_from_loss
from training.optimizer import create_adamw_optimizer
from training.train import TrainConfig, train_model
from training.sft_train import SFTConfig, train_sft_model
from inference.generate import generate_from_prompt, load_model_from_final_state


# Make model directory importable (gpt_model imports local files inside model/).
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.gpt_model import GPT

# Model and data configuration
VOCAB_SIZE = 16_000
SEQ_LENGTH = 256
SEQ_STRIDE = None
LOW_RAM_MODE = False
SEED = 1337
MAX_CPU_THREADS = 12
MAX_CPU_INTEROP_THREADS = 6

# Runtime configuration
DEVICE = "auto"
AMP_DTYPE = "auto"
ALLOW_TF32 = True
CUDNN_BENCHMARK = True
USE_COMPILE = False
DATALOADER_NUM_WORKERS = 0
BATCH_SIZE_CUDA = 32
GRAD_ACCUM_STEPS_CUDA = 2
BATCH_SIZE_CPU = 8
GRAD_ACCUM_STEPS_CPU = 4

# Training configuration
EMBEDDING_DIM = 384
NUM_HEADS = 8
NUM_BLOCKS = 4
DROPOUT = 0.1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
EPOCHS = 3
MAX_STEPS_PER_EPOCH = None
GRAD_CLIP = 1.0
LOG_INTERVAL = 20
WARMUP_RATIO = 0.02
LR_DECAY_FACTOR = 0.1
USE_AMP = True
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_EVERY_STEPS = 500
EVAL_INTERVAL = 1000
EVAL_MAX_BATCHES = 200

# Loss configuration
LOSS_IGNORE_INDEX = -100
LABEL_SMOOTHING = 0.1
Z_LOSS_WEIGHT = 1e-4

# Prepared data and artifacts
PRETRAIN_DATASET_SPLIT = "fr"
PRETRAIN_TOKENIZER_SAMPLE_BYTES = 300 * 1024 * 1024
PRETRAIN_TOKEN_BUDGET = 500_000_000
PRETRAIN_VAL_PERCENT = 1
PRETRAIN_SEED = 42
DATASET_JSONL_PATH = "data/datasets/train.jsonl"
VAL_DATASET_JSONL_PATH = "data/datasets/val.jsonl"
CHECKPOINT_DIR = "artifacts/checkpoints"
CHECKPOINT_NAME = "gpt_latest.pt"
SAVE_FINAL_MODEL_STATE = True
FINAL_MODEL_DIR = "artifacts/models"
FINAL_MODEL_NAME = "gpt_final_state.pt"
TOKENIZER_MODEL_PATH = "tokenizer/token/bpe_model.json"
PREPARED_CACHE_DIR = Path("artifacts/cache")
PIPELINE_MODE = "resume"
ENCODE_CORPUS_LIMIT = 0
MAX_TOKENS_FOR_TRAIN = None
ENCODING_CACHE_PATH = "artifacts/cache/encoded_tokens.json"

# SFT data and training configuration
SFT_DATASET_SPLIT = "train"
SFT_MAX_EXAMPLES = 30_000   #& Nombre maximum d'exemples SFT à utiliser pour l'entrainement (0 = tous)
SFT_VAL_RATIO = 0.01
SFT_MIN_ANSWER_TOKENS = 3
SFT_MAX_ANSWER_TOKENS = 160
SFT_MAX_PROMPT_TOKENS = None
SFT_CONTEXT_MODE = "merge"
SFT_IDENTITY_JSONL_PATH = None
SFT_IDENTITY_REPEAT = 3
SFT_PREPARE_SEED = 426845132   #& Seed pour le mélange aléatoire des exemples SFT lors de la préparation du dataset
SFT_OUT_PREFIX = Path("artifacts/sft/sft")
SFT_TRAIN_PATH = Path("artifacts/sft/sft_train.npz")
SFT_VAL_PATH = Path("artifacts/sft/sft_val.npz")
SFT_META_PATH = Path("artifacts/sft/sft_meta.json")
SFT_CHECKPOINT_DIR = "artifacts/sft/checkpoints"
SFT_CHECKPOINT_NAME = "sft_latest.pt"
SFT_FINAL_MODEL_DIR = "artifacts/sft/models"
SFT_FINAL_MODEL_NAME = "sft_final_state.pt"
SFT_RUNS_DIR = Path("artifacts/sft/runs")
SFT_ACTIVE_RUN_PATH = Path("artifacts/sft/active_run.json")
SFT_LEARNING_RATE = 1e-5
SFT_WEIGHT_DECAY = 0.01
SFT_EPOCHS = 2
SFT_LABEL_SMOOTHING = 0.0
SFT_Z_LOSS_WEIGHT = 0.0
SFT_LABEL_ALIGNMENT_VERSION = 3
SFT_RESUME_FROM_CHECKPOINT = True
SFT_CHECKPOINT_EVERY_STEPS = 100
SFT_EVAL_INTERVAL = 0
SFT_EVAL_MAX_BATCHES = 200

# Inference config
INFER_PROMPT = "bonjour je suis un LLM français nommé Portr-AI."
INFER_MAX_NEW_TOKENS = 512
INFER_TEMPERATURE = 0.9
INFER_TOP_K = 20
ENABLE_INTERACTIVE_REPROMPT = True
INFERENCE_ONLY_FROM_FINAL_MODEL = False


def _setup_device():
    """ Configure le device pour l'entrainement ou l'inference, en tenant compte de la disponibilité CUDA et des préférences de l'utilisateur.

    Raises:
        RuntimeError: Erreur de chargement CUDA si DEVICE demande mais indisponible.

    Returns:
        torch.device: Device configuré pour l'entrainement ou l'inference (CPU ou GPU).
    """
    want_cuda = DEVICE.startswith("cuda") or DEVICE == "auto"
    if want_cuda and not torch.cuda.is_available():
        if torch.version.cuda is None:
            print(f"[cuda] PyTorch CPU-only installe ({torch.__version__}) -> entrainement sur CPU.")
            print("[cuda] Pour utiliser le GPU :")
            print("[cuda]   pip uninstall -y torch")
            print("[cuda]   pip install torch --index-url https://download.pytorch.org/whl/cu128")
        else:
            print(f"[cuda] build CUDA {torch.version.cuda} mais aucun GPU visible (driver ? CUDA_VISIBLE_DEVICES ?)")
        if DEVICE != "auto":
            raise RuntimeError(f"DEVICE={DEVICE} demande mais CUDA indisponible")
        return torch.device("cpu")

    if not want_cuda:
        return torch.device(DEVICE)

    device = torch.device("cuda" if DEVICE == "auto" else DEVICE)
    props = torch.cuda.get_device_properties(device)
    print(
        f"[cuda] {props.name} | VRAM {props.total_memory / 1024**3:.1f} Go | "
        f"sm_{props.major}{props.minor} | CUDA {torch.version.cuda} | "
        f"bf16={'oui' if torch.cuda.is_bf16_supported() else 'non'}"
    )

    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    torch.set_float32_matmul_precision("high" if ALLOW_TF32 else "highest")
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    return device


def _batch_config(device: torch.device):
    """ Retourne la taille du batch et le nombre d'accumulation de gradients selon le device.

    Args:
        device (torch.device): Device cible pour l'entrainement ou l'inference (CPU ou GPU).

    Returns:
        tuple[int, int]: taille du Batch CPU et nombre de step CPU supportable
    """
    if device.type == "cuda":
        return BATCH_SIZE_CUDA, GRAD_ACCUM_STEPS_CUDA
    return BATCH_SIZE_CPU, GRAD_ACCUM_STEPS_CPU


def _validate_files():
    """
    Vérifie l'existence des fichiers essentiels et la validité des modes de cache/tokenizer

    Returns:
        bool: True si tous les fichiers nécessaires sont présents et les modes valides, sinon False
    """
    _assert_no_incomplete_preparation()
    if not Path(DATASET_JSONL_PATH).exists():
        print(f"[main] dataset introuvable: {DATASET_JSONL_PATH}. Lancez prepare_data.py.")
        return False
    if not Path(TOKENIZER_MODEL_PATH).exists():
        print(f"[main] tokenizer introuvable: {TOKENIZER_MODEL_PATH}. Lancez le mode 2.")
        return False
    for split in ("train", "val"):
        manifest = PREPARED_CACHE_DIR / f"{split}_tokens.json"
        if not manifest.exists():
            print(f"[main] cache {split} introuvable: {manifest}. Lancez le mode 2.")
            return False
    return True


def _assert_no_incomplete_preparation():
    if (PREPARED_CACHE_DIR / ".preparation_in_progress").exists():
        raise RuntimeError("Preparation pre-entrainement incomplete. Terminer la preparation (option 2) avant d'utiliser tokenizer et caches.")


def _base_data_ready():
    """Indique si les sorties de préparation du pré-entraînement existent."""
    required = (
        Path(DATASET_JSONL_PATH),
        Path(VAL_DATASET_JSONL_PATH),
        Path(TOKENIZER_MODEL_PATH),
        PREPARED_CACHE_DIR / "train_tokens.json",
        PREPARED_CACHE_DIR / "val_tokens.json"
    )
    return not (PREPARED_CACHE_DIR / ".preparation_in_progress").exists() and all(path.exists() for path in required)


def _sft_data_ready():
    """Indique si les deux splits SFT et leurs métadonnées existent."""
    if (PREPARED_CACHE_DIR / ".preparation_in_progress").exists():
        return False
    if not all(path.exists() for path in (SFT_TRAIN_PATH, SFT_VAL_PATH, SFT_META_PATH)):
        return False
    try:
        metadata = json.loads(SFT_META_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("label_alignment_version") != SFT_LABEL_ALIGNMENT_VERSION:
        return False
    if metadata.get("seq_length") != SEQ_LENGTH or metadata.get("ignore_index") != LOSS_IGNORE_INDEX:
        return False
    special_tokens = metadata.get("special_tokens")
    if not isinstance(special_tokens, dict):
        return False
    try:
        token_to_id = Tokenizer.load(TOKENIZER_MODEL_PATH).get_token_to_id()
        return (
            metadata.get("tokenizer_vocab_size") == len(token_to_id)
            and all(token_to_id.get(token) == token_id for token, token_id in special_tokens.items())
            and set(special_tokens) == {"<|user|>", "<|assistant|>", "<|end|>", "<PAD>"}
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _base_model_path():
    return Path(FINAL_MODEL_DIR) / FINAL_MODEL_NAME


def _sft_model_path():
    run_dir = _active_sft_run()
    return (run_dir / "models" if run_dir else Path(SFT_FINAL_MODEL_DIR)) / SFT_FINAL_MODEL_NAME


def _active_sft_run():
    """Return the explicitly selected run; legacy paths remain the default."""
    if not SFT_ACTIVE_RUN_PATH.exists():
        return None
    try:
        run_dir = Path(json.loads(SFT_ACTIVE_RUN_PATH.read_text(encoding="utf-8"))["run_dir"]).resolve()
        if not run_dir.is_relative_to(SFT_RUNS_DIR.resolve()) or run_dir == SFT_RUNS_DIR.resolve():
            raise ValueError("run en dehors du repertoire SFT")
        return run_dir
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Selection du run SFT invalide: {SFT_ACTIVE_RUN_PATH}") from exc


def _activate_sft_run(run_dir):
    """Select a fresh run before training so an interruption can be resumed."""
    SFT_ACTIVE_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = SFT_ACTIVE_RUN_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps({"run_dir": str(run_dir.resolve())}, indent=2), encoding="utf-8")
    os.replace(temporary_path, SFT_ACTIVE_RUN_PATH)


def _print_project_status():
    """Affiche l'état des sorties principales du projet."""
    print("\n[status] Etat du projet")
    print(f"[status] Donnees pre-entrainement preparees : {_base_data_ready()}")
    print(f"[status] Modele pre-entraine : {_base_model_path().exists()} ({_base_model_path()})")
    print(f"[status] Donnees SFT preparees : {_sft_data_ready()}")
    print(f"[status] Modele SFT entraine : {_sft_model_path().exists()} ({_sft_model_path()})")


def _prepare_pretraining_data() -> None:
    """Lance toute la préparation du corpus de pré-entraînement."""
    import prepare_data

    print("[main] preparation du corpus de pre-entrainement")
    prepare_data.prepare_dataset(
        split=PRETRAIN_DATASET_SPLIT,
        token_budget=PRETRAIN_TOKEN_BUDGET,
        tokenizer_sample_bytes=PRETRAIN_TOKENIZER_SAMPLE_BYTES,
        val_percent=PRETRAIN_VAL_PERCENT,
        seed=PRETRAIN_SEED,
        output_dir=Path(DATASET_JSONL_PATH).parent,
        cache_dir=PREPARED_CACHE_DIR,
        tokenizer_path=Path(TOKENIZER_MODEL_PATH),
        vocab_size=VOCAB_SIZE
    )


def _prepare_sft_data():
    """Lance la préparation du dataset instruction/réponse SFT."""
    import prepare_sft_data

    print("[main] preparation du dataset SFT")
    prepare_sft_data.prepare(argparse.Namespace(
        list_sources=False,
        split=SFT_DATASET_SPLIT,
        tokenizer_path=Path(TOKENIZER_MODEL_PATH),
        seq_length=SEQ_LENGTH,
        max_examples=SFT_MAX_EXAMPLES,
        val_ratio=SFT_VAL_RATIO,
        min_answer_tokens=SFT_MIN_ANSWER_TOKENS,
        max_answer_tokens=SFT_MAX_ANSWER_TOKENS,
        max_prompt_tokens=SFT_MAX_PROMPT_TOKENS,
        context_mode=SFT_CONTEXT_MODE,
        identity_jsonl=Path(SFT_IDENTITY_JSONL_PATH) if SFT_IDENTITY_JSONL_PATH else None,
        identity_repeat=SFT_IDENTITY_REPEAT,
        seed=SFT_PREPARE_SEED,
        out_prefix=SFT_OUT_PREFIX,
        no_update_tokenizer=False
    ))


def _build_sft_dataloaders(device: torch.device):
    """Construit les DataLoaders SFT depuis les fichiers NPZ préparés."""
    if not _sft_data_ready():
        raise RuntimeError("Dataset SFT absent. Lancez d'abord l'action de preparation SFT.")

    train_dataset = SFTDataset(SFT_TRAIN_PATH)
    val_dataset = SFTDataset(SFT_VAL_PATH)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Les splits SFT train et val doivent etre non vides.")
    batch_size, _ = _batch_config(device)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": DATALOADER_NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": DATALOADER_NUM_WORKERS > 0,
    }
    sampler = ResumableRandomSampler(len(train_dataset), seed=SEED)
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(
        f"[main] SFT sequences: train={len(train_dataset)} val={len(val_dataset)} "
        f"batch_size={batch_size}"
    )
    return train_loader, val_loader


def _train_sft_pipeline(device: torch.device, fresh: bool = False):
    """Charge le modèle de base et lance le fine-tuning SFT."""
    _assert_no_incomplete_preparation()
    base_path = _base_model_path()
    if not base_path.exists():
        raise RuntimeError(f"Modele pre-entraine absent: {base_path}")

    tokenizer_model = Tokenizer.load(TOKENIZER_MODEL_PATH)
    model, _payload = load_model_from_final_state(base_path, tokenizer_model, device=str(device))
    tokenizer_ids = tokenizer_model.get_token_to_id().values()
    required_vocab_size = max(len(tokenizer_model.get_token_to_id()), max(tokenizer_ids, default=-1) + 1)
    if required_vocab_size > model.vocab_size:
        model.resize_token_embeddings(required_vocab_size)
        print(f"[main] embeddings SFT redimensionnes: vocab={required_vocab_size}")
    model.to(device)
    train_loader, val_loader = _build_sft_dataloaders(device)
    _, grad_accum_steps = _batch_config(device)
    run_dir = (
        SFT_RUNS_DIR / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        if fresh else _active_sft_run()
    )

    config = SFTConfig(
        epochs=SFT_EPOCHS,
        device=str(device),
        learning_rate=SFT_LEARNING_RATE,
        weight_decay=SFT_WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        log_interval=LOG_INTERVAL,
        checkpoint_dir=str(run_dir / "checkpoints") if run_dir else SFT_CHECKPOINT_DIR,
        checkpoint_name=SFT_CHECKPOINT_NAME,
        resume_from_checkpoint=SFT_RESUME_FROM_CHECKPOINT and not fresh,
        checkpoint_every_steps=SFT_CHECKPOINT_EVERY_STEPS,
        save_final_model_state=True,
        final_model_dir=str(run_dir / "models") if run_dir else SFT_FINAL_MODEL_DIR,
        final_model_name=SFT_FINAL_MODEL_NAME,
        warmup_ratio=WARMUP_RATIO,
        lr_decay_factor=LR_DECAY_FACTOR,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        grad_accum_steps=grad_accum_steps,
        max_cpu_threads=MAX_CPU_THREADS,
        max_cpu_interop_threads=MAX_CPU_INTEROP_THREADS,
        seed=SEED,
        eval_interval=SFT_EVAL_INTERVAL,
        eval_max_batches=SFT_EVAL_MAX_BATCHES,
        benchmark_path=str(run_dir / "training_benchmark.csv") if run_dir else "artifacts/benchmarks/sft_training_benchmark.csv",
        label_smoothing=SFT_LABEL_SMOOTHING,
        z_loss_weight=SFT_Z_LOSS_WEIGHT,
        compile_model=False
    )
    if fresh:
        _activate_sft_run(run_dir)
        print(f"[main] nouveau run SFT isole: {run_dir}; reprise=6, inference=8")
    history = train_sft_model(model, train_loader, config, val_dataloader=val_loader)
    print(f"[main] SFT checkpoint path: {history['checkpoint_path']}")
    print(f"[main] SFT final model path: {history['final_model_path']}")
    return model, history


def _select_pipeline_mode_console():
    """Choisit une action du manager de projet."""
    global PIPELINE_MODE, RESUME_FROM_CHECKPOINT

    options = {
        "1": ("status", "Afficher l'etat des etapes"),
        "2": ("prepare_pretraining", "Preparer les donnees pre-entrainement"),
        "3": ("train_pretraining", "Entrainer / reprendre le modele de base"),
        "4": ("train_pretraining_fresh", "Entrainer un modele de base neuf"),
        "5": ("prepare_sft", "Preparer les donnees SFT"),
        "6": ("train_sft", "Entrainer le modele SFT"),
        "7": ("inference_pretraining", "Tester le modele de base"),
        "8": ("inference_sft", "Tester le modele SFT"),
        "9": ("full_pretraining", "Executer le pipeline pre-entrainement complet"),
        "10": ("train_sft_fresh", "Repartir du modele de base pour un nouveau SFT"),
    }
    print("\n+-----------------------------------------------------------+")
    print("|                 Manager du projet Portr-AI                |")
    print("+-----------------------------------------------------------+")
    for key, (_, label) in options.items():
        print(f"|  {key}  {label:<51}   |")
    print("+-----------------------------------------------------------+")

    try:
        choice = input("[main] Choisir une action [1-10] (Entree = 1): ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\n[main] choix interrompu")
        return None

    selected = options.get(choice)
    if selected is None:
        print(f"[main] choix invalide: {choice}")
        return None

    PIPELINE_MODE = selected[0]
    RESUME_FROM_CHECKPOINT = PIPELINE_MODE != "train_pretraining_fresh"
    print(f"[main] mode selectionne: {selected[1]}")
    return PIPELINE_MODE


def _file_signature(path: str) -> dict:
    """
    Fournit une signature de fichier simple pour vérifier l'existence et les métadonnées de base (taille, mtime)

    Args:
        path (str): Chemin du fichier à vérifier

    Returns:
        dict: Dictionnaire contenant:
            - path (str): Chemin du fichier vérifié
            - exists (bool): Indique si le fichier existe
            - size (int, optionnel): Taille du fichier en octets (si existe)
    """
    p = Path(path)
    if not p.exists():
        return {"path": path, "exists": False}
    st = p.stat()
    return {
        "path": path,
        "exists": True,
        "size": int(st.st_size),
        "mtime": float(st.st_mtime)
    }


def _build_encoding_meta(tokenizer_model) -> dict:
    """
    Construit un dictionnaire de métadonnées décrivant les conditions d'encodage du corpus, 
    utilisé pour valider la pertinence du cache d'encodage existant et pour décider 
    s'il faut ré-encoder le corpus ou utiliser le cache.

    Args:
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour déterminer la taille du vocabulaire et les métadonnées d'encodage, ainsi que pour valider le cache existant et potentiellement ré-encoder le corpus si nécessaire.

    Returns:
        dict: disctionnaire de métadonnées
    """
    return {
        "dataset": _file_signature(DATASET_JSONL_PATH),
        "tokenizer_model": _file_signature(TOKENIZER_MODEL_PATH),
        "target_vocab_size": VOCAB_SIZE,
        "encode_corpus_limit": int(ENCODE_CORPUS_LIMIT),
        "max_tokens_for_train": MAX_TOKENS_FOR_TRAIN,
        "tokenizer_vocab_size": int(len(tokenizer_model.get_token_to_id()))
    }


def _save_token_cache(tokens: array, meta: dict):
    """
    Sauvegarde les tokens encodés dans un format binaire compact et les métadonnées associées pour une validation future.

    Args:
        tokens (array): array('i') des tokens encodés (int32).
        meta (dict): Dictionnaire de métadonnées décrivant les conditions d'encodage (dataset, tokenizer, paramètres) pour valider la pertinence du cache lors du chargement.
    """
    cache_path = Path(ENCODING_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    #& Stockage binaire compact (4 octets par token) pour limiter la RAM
    #& au chargement via torch.from_file (memory-mapped).
    bin_path = cache_path.with_suffix(".bin")
    assert tokens.itemsize == 4, "array('i') doit faire 4 octets par element"
    with bin_path.open("wb") as f:
        tokens.tofile(f)  #& ecriture directe du buffer, sans conversion Python

    payload = {
        "meta": meta,
        "token_count": len(tokens),
        "dtype": "int32",
        "bin_path": str(bin_path)
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[main] cache ecrit: {cache_path} + {bin_path} ({len(tokens)} tokens)")


def _load_token_cache(expected_meta: dict):
    """
    Charge les tokens encodés à partir du cache si celui-ci est valide en comparant les métadonnées d'encodage attendues avec 
    celles stockées dans le cache. Si le cache est absent, invalide ou ne correspond pas aux métadonnées attendues, 
    retourne None pour indiquer qu'un nouvel encodage doit être effectué.

    Args:
        expected_meta (dict): Dictionnaire de métadonnées décrivant les conditions d'encodage attendues (dataset, tokenizer, paramètres) pour valider la pertinence du cache existant.

    Returns:
        torch.Tensor | None: Tensor de tokens encodés chargé à partir du cache si valide, 
        sinon None pour indiquer qu'un nouvel encodage doit être effectué. 
        Le tensor est chargé de manière memory-mapped pour limiter l'utilisation de RAM
        lors du chargement de grands fichiers de tokens.
    """
    cache_path = Path(ENCODING_CACHE_PATH)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    meta = payload.get("meta")
    token_count = payload.get("token_count")
    bin_path_raw = payload.get("bin_path")
    if not isinstance(meta, dict):
        return None
    if not isinstance(token_count, int) or token_count <= 0:
        return None
    if not isinstance(bin_path_raw, str) or not bin_path_raw:
        return None

    if meta != expected_meta:
        return None

    bin_path = Path(bin_path_raw)
    if not bin_path.exists():
        return None

    expected_bytes = token_count * 4
    if bin_path.stat().st_size != expected_bytes:
        return None

    return torch.from_file(
        str(bin_path), shared=False,
        size=token_count, dtype=torch.int32)


def _get_or_encode_tokens(tokenizer_model):
    """
    Obtient les tokens encodés à partir du cache si disponible et valide, sinon encode le corpus et sauvegarde le cache pour les futurs runs

    Args:
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour déterminer la taille du vocabulaire et les métadonnées d'encodage, ainsi que pour valider le cache existant et potentiellement ré-encoder le corpus si nécessaire.

    Returns:
        torch.Tensor: Tensor int32 de tokens encodés (memory-mapped si issu du cache).
    """
    _assert_no_incomplete_preparation()
    manifest = PREPARED_CACHE_DIR / "train_tokens.json"
    with manifest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    bin_path = Path(payload["bin_path"])
    token_count = int(payload["token_count"])
    if not bin_path.exists() or bin_path.stat().st_size != token_count * 4:
        raise RuntimeError(f"Cache train invalide: {manifest}. Relancez le mode 2.")
    return torch.from_file(str(bin_path), shared=False, size=token_count, dtype=torch.int32)


def _check_dataset_quality(path: str, sample_lines: int = 5000):
    """
    Vérifie la qualité du dataset en échantillonnant les premières lignes du fichier JSONL

    Args:
        path (str): chemin du fichier dataset.jsonl à vérifier
        sample_lines (int, optional): Emplacement de l'échantillon. Defaults to 5000.

    Returns:
        bool: True si le dataset semble de qualité suffisante pour l'entraînement, sinon False
    """
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


def _build_dataloaders(tokenizer_model, device: torch.device):
    """
    Construit les DataLoaders train / validation à partir des tokens encodés.

    - Validation = les derniers VAL_FRACTION % des tokens (pas de fuite :
      les fenêtres train et val ne se chevauchent pas).
    - Sampler reproductible (seed + epoch) -> reprise exacte en milieu d'epoch.

    Args:
        tokenizer_model: Tokenizer déjà chargé.
        device: Device d'entraînement (pour batch size et pin_memory).

    Raises:
        ValueError: Pour éviter les erreurs d'entrainements

    Returns:
        tuple[DataLoader, DataLoader | None]: (train, validation ou None)
    """
    tokens = _get_or_encode_tokens(tokenizer_model)
    n_tokens = int(tokens.numel())
    print(f"[main] nb tokens encodes: {n_tokens}")

    if n_tokens <= SEQ_LENGTH + 1:
        raise ValueError(f"Pas assez de tokens pour SEQ_LENGTH={SEQ_LENGTH}. tokens={n_tokens}")

    val_manifest = PREPARED_CACHE_DIR / "val_tokens.json"
    val_tokens = None
    if val_manifest.exists():
        with val_manifest.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        bin_path = Path(payload["bin_path"])
        token_count = int(payload["token_count"])
        if bin_path.exists() and bin_path.stat().st_size == token_count * 4:
            val_tokens = torch.from_file(str(bin_path), shared=False, size=token_count, dtype=torch.int32)

    train_ds = TextDataset(tokens, SEQ_LENGTH, stride=SEQ_STRIDE)
    val_ds = TextDataset(val_tokens, SEQ_LENGTH) if val_tokens is not None else None

    batch_size, _ = _batch_config(device)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=DATALOADER_NUM_WORKERS > 0
    )

    sampler = ResumableRandomSampler(len(train_ds), seed=SEED, replacement=LOW_RAM_MODE)
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs) if val_ds is not None else None

    print(f"[main] train sequences: {len(train_ds)} | val sequences: {len(val_ds) if val_ds else 0} | batch_size={batch_size} | micro-batches/epoch={len(train_loader)}")
    x0, y0 = train_ds[0]
    print(f"[main] sample shapes: input={tuple(x0.shape)} target={tuple(y0.shape)}")
    return train_loader, val_loader


def _train_pipeline(dataloader: DataLoader, val_dataloader: DataLoader | None, tokenizer_model, device: torch.device):
    """
    Entraîne le modèle GPT sur les données fournies par le dataloader, en utilisant les configurations définies et en sauvegardant les checkpoints et le modèle final

    Args:
        dataloader (DataLoader): Dataloader fournissant les batches d'entraînement
        val_dataloader (DataLoader | None): Dataloader de validation
        device (torch.device): Device d'entraînement
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour déterminer la taille du vocabulaire et les métadonnées d'encodage

    Returns:
        tuple[GPT, dict]: Le modèle GPT entraîné et un dictionnaire d'historique contenant les pertes par époque, les chemins des checkpoints et du modèle final, et d'autres métriques pertinentes pour l'analyse post-entraînement.
    """
    vocab_size = max(tokenizer_model.get_token_to_id().values()) + 1
    checkpoint_path = Path(CHECKPOINT_DIR) / CHECKPOINT_NAME
    if RESUME_FROM_CHECKPOINT and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        saved_vocab = payload["model_state_dict"]["embedding.token_emb.weight"].shape[0]
        del payload
        if saved_vocab != vocab_size:
            raise RuntimeError(
                f"Reprise de base incompatible: checkpoint vocab={saved_vocab}, "
                f"tokenizer vocab={vocab_size}. Ne pas redimensionner silencieusement "
                "un optimizer sauvegarde. Verifier les IDs du tokenizer et choisir "
                "une migration explicite ou un nouveau run."
            )
    print(f"[main] vocab_size effective: {vocab_size}")

    model = GPT(
        vocab_size=vocab_size,
        seq_length=SEQ_LENGTH,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=DROPOUT
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[main] parametres du modele: {n_params / 1e6:.2f}M")

    #& Le modele doit etre sur le device AVANT l'optimizer (AdamW fused = CUDA)
    model.to(device)
    optimizer = create_adamw_optimizer(model=model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    _, grad_accum_steps = _batch_config(device)
    train_cfg = TrainConfig(
        epochs=EPOCHS,
        device=str(device),
        grad_clip=GRAD_CLIP,
        log_interval=LOG_INTERVAL,
        max_steps_per_epoch=MAX_STEPS_PER_EPOCH,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
        checkpoint_every_steps=CHECKPOINT_EVERY_STEPS,
        save_final_model_state=SAVE_FINAL_MODEL_STATE,
        final_model_dir=FINAL_MODEL_DIR,
        final_model_name=FINAL_MODEL_NAME,
        warmup_ratio=WARMUP_RATIO,
        lr_decay_factor=LR_DECAY_FACTOR,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        grad_accum_steps=grad_accum_steps,
        max_cpu_threads=MAX_CPU_THREADS,
        max_cpu_interop_threads=MAX_CPU_INTEROP_THREADS,
        seed=SEED,
        eval_interval=EVAL_INTERVAL,
        eval_max_batches=EVAL_MAX_BATCHES,
        compile_model=USE_COMPILE
    )

    loss_fn = partial(autoregressive_cross_entropy, ignore_index=LOSS_IGNORE_INDEX,
        label_smoothing=LABEL_SMOOTHING, z_loss_weight=Z_LOSS_WEIGHT)

    #& Validation : cross-entropy pure (sans label smoothing ni z-loss)
    #& -> perplexite interpretable
    eval_loss_fn = partial(autoregressive_cross_entropy, ignore_index=LOSS_IGNORE_INDEX)

    history = train_model(model=model, dataloader=dataloader,
        optimizer=optimizer, loss_fn=loss_fn, config=train_cfg,
        val_dataloader=val_dataloader, eval_loss_fn=eval_loss_fn)

    if history["epoch_losses"]:
        print(f"[main] final train loss (avec label smoothing): {history['epoch_losses'][-1]:.4f}")
    if history["val_losses"]:
        last_val = history["val_losses"][-1]
        print(f"[main] final val loss: {last_val:.4f} | val perplexity: {perplexity_from_loss(last_val):.2f}")

    return model, history


def _run_inference(model: GPT, tokenizer_model, device: torch.device, use_sft_template: bool):
    """
    Fonction d'inférence qui génère du texte à partir d'un prompt donné, en utilisant le modèle GPT entraîné et le tokenizer pour décoder les tokens générés en texte lisible

    Args:
        model (GPT): Modèle GPT entraîné à utiliser pour l'inférence
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour décoder les tokens générés en texte lisible
        device (torch.device): Device d'inférence
        use_sft_template (bool): Utilise le format instruction/réponse SFT.
    """
    model.to(device)

    token_ids, text = generate_from_prompt(
        model=model,
        tokenizer=tokenizer_model,
        prompt=INFER_PROMPT,
        max_new_tokens=INFER_MAX_NEW_TOKENS,
        temperature=INFER_TEMPERATURE,
        top_k=INFER_TOP_K,
        device=device,
        use_sft_template=use_sft_template
    )

    print("[main] inference done")
    print(f"[main] generated token count: {len(token_ids)}")
    print(f"[main] generated text:\n{text}")


def _load_model_for_inference_from_artifact(tokenizer_model, final_model_path: str, device: torch.device):
    """Charge un modele GPT final pour inference sans repasser par training.

    Args:
        tokenizer_model: Tokenizer deja charge.
        final_model_path (str): Chemin du modele final sauvegarde.

    Returns:
        GPT | None: Modele charge si disponible, sinon None.
    """
    model_path = Path(final_model_path)
    if not model_path.exists():
        print(f"[main] final model not found: {model_path}")
        return None

    model, _payload = load_model_from_final_state(
        final_model_path=model_path,
        tokenizer=tokenizer_model, device=device)
    tokenizer_ids = tokenizer_model.get_token_to_id().values()
    required_vocab_size = max(len(tokenizer_model.get_token_to_id()), max(tokenizer_ids, default=-1) + 1)
    if required_vocab_size > model.vocab_size:
        model.resize_token_embeddings(required_vocab_size)
        print(f"[main] embeddings redimensionnes: vocab={required_vocab_size}")
    print(f"[main] final model loaded for inference: {model_path}")
    return model


def _interactive_reprompt_loop(model: GPT, tokenizer_model, device: torch.device, use_sft_template: bool):
    """ Boucle de reprompt du modèle

    Args:
        model (GPT): Modèle GPT entraîné à utiliser pour l'inférence
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour décoder les tokens générés en texte lisible
        device (torch.device): Device d'inférence
        use_sft_template (bool): Utilise le format instruction/réponse SFT.
    """
    if not ENABLE_INTERACTIVE_REPROMPT:
        return

    print("\n[main] mode reprompt actif")
    print("[main] Tape un prompt puis Entrer.")
    print("[main] Commandes: ':quit' pour sortir, ':help' pour aide.")

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

        #& Une erreur sur un prompt ne doit pas tuer toute la session
        try:
            token_ids, text = generate_from_prompt(
                model=model,
                tokenizer=tokenizer_model,
                prompt=user_input,
                max_new_tokens=INFER_MAX_NEW_TOKENS,
                temperature=INFER_TEMPERATURE,
                top_k=INFER_TOP_K,
                device=device,
                use_sft_template=use_sft_template
            )
        except Exception as exc:
            print(f"[reprompt] erreur: {exc}")
            continue

        print(f"[reprompt] tokens: {len(token_ids)}")
        print(f"[reprompt] text:\n{text}\n")


# *=====================================================================================================================*
# *=====================================================================================================================*
# *=====================================================================================================================*

def _train_pretraining_action(device: torch.device, fresh: bool = False):
    """Exécute l'entraînement pré-entraînement avec ou sans reprise."""
    global RESUME_FROM_CHECKPOINT

    if not _validate_files():
        return None
    if not _check_dataset_quality(DATASET_JSONL_PATH):
        print("[main] controle qualite du dataset echoue")
        return None

    RESUME_FROM_CHECKPOINT = not fresh
    tokenizer_model = Tokenizer.load(TOKENIZER_MODEL_PATH)
    train_loader, val_loader = _build_dataloaders(tokenizer_model, device)
    model, history = _train_pipeline(train_loader, val_loader, tokenizer_model, device)
    print(f"[main] checkpoint path: {history.get('checkpoint_path')}")
    print(f"[main] final model path: {history.get('final_model_path')}")
    return model


def _inference_action(device: torch.device, sft: bool):
    """Charge et teste le modèle de base ou le modèle SFT."""
    _assert_no_incomplete_preparation()
    tokenizer_model = Tokenizer.load(TOKENIZER_MODEL_PATH)
    model_path = _sft_model_path() if sft else _base_model_path()
    model = _load_model_for_inference_from_artifact(tokenizer_model, str(model_path), device)
    if model is None:
        return

    _run_inference(model, tokenizer_model, device, use_sft_template=sft)
    _interactive_reprompt_loop(model, tokenizer_model, device, use_sft_template=sft)


def main() -> None:
    """Lance le manager interactif des étapes du projet."""
    os.chdir(ROOT_DIR)
    print("[main] manager Portr-AI")
    action = _select_pipeline_mode_console()
    if action is None:
        return
    if action == "status":
        _print_project_status()
        return

    device = _setup_device()

    try:
        if action == "prepare_pretraining":
            _prepare_pretraining_data()
        elif action == "prepare_sft":
            if not Path(TOKENIZER_MODEL_PATH).exists():
                raise RuntimeError("Tokenizer absent. Preparez d'abord les donnees pre-entrainement.")
            _prepare_sft_data()
        elif action == "train_pretraining":
            _train_pretraining_action(device)
        elif action == "train_pretraining_fresh":
            _train_pretraining_action(device, fresh=True)
        elif action == "train_sft":
            _train_sft_pipeline(device)
        elif action == "train_sft_fresh":
            _train_sft_pipeline(device, fresh=True)
        elif action == "inference_pretraining":
            _inference_action(device, sft=False)
        elif action == "inference_sft":
            _inference_action(device, sft=True)
        elif action == "full_pretraining":
            if not _base_data_ready():
                _prepare_pretraining_data()
            if INFERENCE_ONLY_FROM_FINAL_MODEL:
                _inference_action(device, sft=False)
            else:
                model = _train_pretraining_action(device)
                if model is not None:
                    _inference_action(device, sft=False)
        print(f"[main] action terminee: {action}")
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"[main] action impossible: {exc}")


if __name__ == "__main__":
    main()
