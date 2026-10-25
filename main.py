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
from pathlib import Path
import sys
from array import array
from collections import Counter
from functools import partial

import torch
from torch.utils.data import DataLoader

from tokenizer.tokenizer import Tokenizer
import prepare_data
from data.dataset import TextDataset, ResumableRandomSampler
from training.loss import autoregressive_cross_entropy, perplexity_from_loss
from training.optimizer import create_adamw_optimizer
from training.train import TrainConfig, train_model
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
    if not Path(DATASET_JSONL_PATH).exists():
        print(f"[main] dataset introuvable: {DATASET_JSONL_PATH}. Lancez prepare_data.py.")
        return False
    if not Path(TOKENIZER_MODEL_PATH).exists():
        print(f"[main] tokenizer introuvable: {TOKENIZER_MODEL_PATH}. Lancez le mode 3.")
        return False
    for split in ("train", "val"):
        manifest = PREPARED_CACHE_DIR / f"{split}_tokens.json"
        if not manifest.exists():
            print(f"[main] cache {split} introuvable: {manifest}. Lancez le mode 3.")
            return False
    return True


def _select_pipeline_mode_console():
    """Choisit le niveau de recalcul des donnees et du modele."""
    global PIPELINE_MODE, RESUME_FROM_CHECKPOINT

    options = {
        "1": ("resume", "Reprendre avec tokenizer et caches prepares"),
        "2": ("fresh_model", "Nouveau modele avec les donnees deja preparees"),
        "3": ("rebuild_all", "Recreer donnees + tokenizer + modele neuf"),
    }
    print("\n+-----------------------------------------------------------+")
    print("|                    Pipeline Portr-AI                    |")
    print("+-----------------------------------------------------------+")
    for key, (_, label) in options.items():
        print(f"|  {key}  {label:<51}|")
    print("+-----------------------------------------------------------+")

    try:
        choice = input("[main] Choisir [1/2/3] (Entree = 1): ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print("\n[main] choix interrompu")
        return False

    selected = options.get(choice)
    if selected is None:
        print(f"[main] choix invalide: {choice}")
        return False

    PIPELINE_MODE = selected[0]
    RESUME_FROM_CHECKPOINT = PIPELINE_MODE == "resume"
    print(f"[main] mode selectionne: {selected[1]}")
    return True


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
    manifest = PREPARED_CACHE_DIR / "train_tokens.json"
    with manifest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    bin_path = Path(payload["bin_path"])
    token_count = int(payload["token_count"])
    if not bin_path.exists() or bin_path.stat().st_size != token_count * 4:
        raise RuntimeError(f"Cache train invalide: {manifest}. Relancez le mode 3.")
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
    vocab_size = len(tokenizer_model.get_token_to_id())
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


def _run_inference(model: GPT, tokenizer_model, device: torch.device):
    """
    Fonction d'inférence qui génère du texte à partir d'un prompt donné, en utilisant le modèle GPT entraîné et le tokenizer pour décoder les tokens générés en texte lisible

    Args:
        model (GPT): Modèle GPT entraîné à utiliser pour l'inférence
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour décoder les tokens générés en texte lisible
        device (torch.device): Device d'inférence
    """
    model.to(device)

    token_ids, text = generate_from_prompt(
        model=model,
        tokenizer=tokenizer_model,
        prompt=INFER_PROMPT,
        max_new_tokens=INFER_MAX_NEW_TOKENS,
        temperature=INFER_TEMPERATURE,
        top_k=INFER_TOP_K,
        device=device
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
    print(f"[main] final model loaded for inference: {model_path}")
    return model


def _interactive_reprompt_loop(model: GPT, tokenizer_model, device: torch.device):
    """ Boucle de reprompt du modèle

    Args:
        model (GPT): Modèle GPT entraîné à utiliser pour l'inférence
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour décoder les tokens générés en texte lisible
        device (torch.device): Device d'inférence
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
            )
        except Exception as exc:
            print(f"[reprompt] erreur: {exc}")
            continue

        print(f"[reprompt] tokens: {len(token_ids)}")
        print(f"[reprompt] text:\n{text}\n")


# *=====================================================================================================================*
# *=====================================================================================================================*
# *=====================================================================================================================*

def main() -> None:
    """
    Fonction principale qui va gérer la pipeline complète du modèle
    """
    print("[main] pipeline start")
    device = _setup_device()

    if not _select_pipeline_mode_console():
        return

    if PIPELINE_MODE == "rebuild_all":
        print("[main:1/6] preparation Fineweb2-HQ en cours...")
        prepare_data.prepare_dataset(
            vocab_size=VOCAB_SIZE,
            output_dir=Path(DATASET_JSONL_PATH).parent,
            cache_dir=PREPARED_CACHE_DIR,
            tokenizer_path=Path(TOKENIZER_MODEL_PATH),
        )

    if not _validate_files():
        return
    print("[main:2/6] fichiers train/val, tokenizer et manifests valides")

    if not _check_dataset_quality(DATASET_JSONL_PATH):
        print("[main] dataset quality check echoue, pipeline stop")
        return
    print("[main:3/6] controle qualite du dataset termine")

    #* 1) Charger le tokenizer et les caches produits par prepare_data.py
    print("[main:4/6] chargement du tokenizer prepare")
    try:
        tokenizer_model = Tokenizer.load(TOKENIZER_MODEL_PATH)
    except FileNotFoundError:
        print("[main] tokenizer introuvable. Lancez le mode 3.")
        return

    #* 2) Build dataloader from prepared encoded corpus
    if INFERENCE_ONLY_FROM_FINAL_MODEL:
        final_model_path = str(Path(FINAL_MODEL_DIR) / FINAL_MODEL_NAME)
        model = _load_model_for_inference_from_artifact(tokenizer_model, final_model_path, device)
        if model is None:
            print("[main] impossible de charger le modele final, pipeline stop")
            return

        _run_inference(model, tokenizer_model, device)
        _interactive_reprompt_loop(model, tokenizer_model, device)
        print("[main] pipeline complete (inference-only mode)")
        return

    train_loader, val_loader = _build_dataloaders(tokenizer_model, device)
    print("[main:5/6] dataloaders train/validation prets")

    #* 3) Train GPT
    print("[main:6/6] donnees preparees chargees; demarrage de l'entrainement GPT...")
    model, history = _train_pipeline(train_loader, val_loader, tokenizer_model, device)

    #* 4) Checkpoint info
    checkpoint_path = history.get("checkpoint_path")
    final_model_path = history.get("final_model_path")
    print(f"[main] checkpoint path: {checkpoint_path}")
    print(f"[main] final model path: {final_model_path}")

    reloaded_model = _load_model_for_inference_from_artifact(tokenizer_model, final_model_path, device)
    if reloaded_model is not None:
        model = reloaded_model

    #* 5) Inference test
    _run_inference(model, tokenizer_model, device)

    #* 6) Interactive reprompt tests
    _interactive_reprompt_loop(model, tokenizer_model, device)

    print("[main] pipeline complete")


if __name__ == "__main__":
    main()
