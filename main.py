"""
main.py

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
from torch.utils.data import DataLoader, RandomSampler

import tokenizer.TokenizeData as td
from data.dataset import TextDataset
from training.loss import autoregressive_cross_entropy, perplexity_from_loss
from training.optimizer import create_adamw_optimizer
from training.train import TrainConfig, train_model
from inference.generate import generate_from_prompt, load_model_from_final_state


# Make model directory importable (gpt_model imports local files inside model/).
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.gpt_model import GPT


# -----------------------------------------------------------------------------
# Variable de configuration des donnees et du modele
# -----------------------------------------------------------------------------
td.VOCAB_SIZE = 30_000  # 8k-16k minimum recommande pour du francais
td.LIMIT_CORPUS = td.VOCAB_SIZE * 10  #! pas de limite pour un entrainement opti

# Variable de configuration des sequences
SEQ_LENGTH = 256
BATCH_SIZE = 8
LOW_RAM_MODE = False #! pas de biais mais reisque de crash sur gros dataset (replacement=True dans sampler)
MAX_TOKENS_FOR_TRAIN = None #! Passer en None pour aucun limite (mais risque de RAM)
DATALOADER_NUM_WORKERS = 0
DATALOADER_PIN_MEMORY = False
MAX_CPU_THREADS = 12
MAX_CPU_INTEROP_THREADS = 6

#* Training config
EMBEDDING_DIM = 384
NUM_HEADS = 8
NUM_BLOCKS = 4
DROPOUT = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 3
MAX_STEPS_PER_EPOCH = None  #! Retirer la limite pour qu'il trouve un équilibre naturel
GRAD_CLIP = 1.0
LOG_INTERVAL = 20
global WARMUP_STEPS #= 2000 #! Calculé dans la fonction _build_dataloader
LR_DECAY_FACTOR = 0.1
USE_AMP = True
GRAD_ACCUM_STEPS = 4    #! était 1 — batch effectif = 8x4 = 32, plus stable
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_EVERY_STEPS = 500

#* Loss config
LOSS_IGNORE_INDEX = -100
LABEL_SMOOTHING = 0.1   #! était 0.0 — réduit l'overconfidence, meilleure généralisation
Z_LOSS_WEIGHT = 1e-4    #! était 0.0 — stabilise les logits

#* Pipeline controls
RUN_TOKENIZER_TRAINING = True
#& Tokenizer mode:
#& - "auto": charge modèle existant sinon entraine
#& - "use_existing": utilise uniquement le modèle deja entrainé
#& - "retrain": force le ré-entraînement
TOKENIZER_MODE = "auto"
DATASET_JSONL_PATH = "data/datasets/dataset.jsonl"
CHECKPOINT_DIR = "artifacts/checkpoints"
CHECKPOINT_NAME = "gpt_latest.pt"
SAVE_FINAL_MODEL_STATE = True #! Sauvegarde un artefact final du modèle pour inference réutilisable (plus léger que checkpoint complet)
FINAL_MODEL_DIR = "artifacts/models"
FINAL_MODEL_NAME = "gpt_final_state.pt"
TOKENIZER_MODEL_PATH = "tokenizer/token/bpe_model.json"

#* Cache controls for heavy preprocessing
#& Modes:
#& - "auto": use cache when valid, otherwise recompute and refresh cache
#& - "use_precomputed": use cache only (fail if missing/invalid)
#& - "recompute": always recompute and overwrite cache
ENCODING_CACHE_MODE = "auto"
ENCODING_CACHE_PATH = "artifacts/cache/encoded_tokens.json"

# Inference config
INFER_PROMPT = "bonjour je suis un LLM français nommé Portr-AI."
INFER_MAX_NEW_TOKENS = 40
INFER_TEMPERATURE = 0.9
INFER_TOP_K = 20
ENABLE_INTERACTIVE_REPROMPT = True
INFERENCE_ONLY_FROM_FINAL_MODEL = False


def _select_cache_mode_console() -> None:
    """
    Laisse l'utilisateur choisir le mode de gestion du cache d'encodage via la console.

    - use_precomputed: utilise le cache existant uniquement (échoue si cache invalide ou manquant)
    - recompute: force le recalcul de l'encodage et écrase le cache existant
    - auto: utilise le cache s'il est valide, sinon recalcul et mise à jour du cache
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
    """
    Vérifie l'existence des fichiers essentiels et la validité des modes de cache/tokenizer

    Returns:
        bool: True si tous les fichiers nécessaires sont présents et les modes valides, sinon False
    """
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
    """
    Permet à l'utilisateur de choisir le mode de gestion du tokenizer via la console
    - use_existing: utilise le tokenizer déjà entraîné (échoue si absent)
    - retrain: force le ré-entraînement du tokenizer
    - auto: utilise le tokenizer existant s'il est présent, sinon entraîne un nouveau tokenizer
    """
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
        "target_vocab_size": int(td.VOCAB_SIZE),
        "corpus_limit": int(td.LIMIT_CORPUS),
        "tokenizer_vocab_size": int(len(tokenizer_model.get_token_to_id())),
        "seq_length": int(SEQ_LENGTH)
    }


def _save_token_cache(tokens: list[int], meta: dict) -> None:
    """
    Sauvegarde les tokens encodés dans un format binaire compact et les métadonnées associées pour une validation future.

    Args:
        tokens (list[int]): Liste des tokens encodés à sauvegarder.
        meta (dict): Dictionnaire de métadonnées décrivant les conditions d'encodage (dataset, tokenizer, paramètres) pour valider la pertinence du cache lors du chargement.
    """
    cache_path = Path(ENCODING_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    #& Stockage binaire compact (4 octets par token) pour limiter la RAM
    #& au chargement via torch.from_file (memory-mapped).
    bin_path = cache_path.with_suffix(".bin")
    with bin_path.open("wb") as f:
        array("I", (int(t) for t in tokens)).tofile(f)

    payload = {
        "meta": meta,
        "token_count": len(tokens),
        "dtype": "uint32",
        "bin_path": str(bin_path)
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[main] cache ecrit: {cache_path} + {bin_path} ({len(tokens)} tokens)")


def _load_token_cache(expected_meta: dict) -> torch.Tensor | None:
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


def _get_or_encode_tokens(tokenizer_model) -> torch.Tensor | list[int]:
    """
    Obtient les tokens encodés à partir du cache si disponible et valide, sinon encode le corpus et sauvegarde le cache pour les futurs runs

    Args:
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour déterminer la taille du vocabulaire et les métadonnées d'encodage, ainsi que pour valider le cache existant et potentiellement ré-encoder le corpus si nécessaire.

    Returns:
        torch.Tensor | list[int]: Tensor de tokens encodés prêt à être utilisé pour le dataset, ou liste de tokens si le cache n'est pas utilisé (dans ce cas, la conversion en tensor se fera plus tard pour éviter de charger tout en RAM d'un coup).
    """
    expected_meta = _build_encoding_meta(tokenizer_model)

    if ENCODING_CACHE_MODE in {"auto", "use_precomputed"}:
        cached_tokens = _load_token_cache(expected_meta)
        if cached_tokens is not None:
            print("[main] cache valide trouve, encodage saute")
            return cached_tokens

        if ENCODING_CACHE_MODE == "use_precomputed":
            print("[main] WARNING: mode use_precomputed actif, mais aucun cache valide n'a ete trouve. Fallback automatique vers recompute pour eviter l'arret.")

    print("[main] encodage du corpus (calcul lourd)...")
    tokens = tokenizer_model.encode_corpus(DATASET_JSONL_PATH, limit=td.LIMIT_CORPUS)

    if MAX_TOKENS_FOR_TRAIN and len(tokens) > MAX_TOKENS_FOR_TRAIN:
        print(f"[main] tokens limites pour RAM: {len(tokens)} -> {MAX_TOKENS_FOR_TRAIN}")
        tokens = tokens[:MAX_TOKENS_FOR_TRAIN]

    _save_token_cache(tokens, expected_meta)
    return torch.tensor(tokens, dtype=torch.int32)


def _check_dataset_quality(path: str, sample_lines: int = 5000) -> bool:
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


def _build_dataloader(tokenizer_model) -> tuple[DataLoader, int]:
    """
    Construit un DataLoader PyTorch à partir des tokens encodés, en utilisant les configurations définies et en gérant les modes de cache pour optimiser le processus d'encodage

    Args:
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour déterminer la taille du vocabulaire et les métadonnées d'encodage, ainsi que pour valider le cache existant et potentiellement ré-encoder le corpus si nécessaire.

    Raises:
        ValueError: Pour éviter les erreurs d'entrainements

    Returns:
        tuple[DataLoader, int]: Le DataLoader prêt pour l'entraînement du modèle GPT, et le nombre total de tokens encodés (utile pour le logging et le calcul des steps par epoch).
    """
    tokens = _get_or_encode_tokens(tokenizer_model)
    n_tokens = int(tokens.numel()) if isinstance(tokens, torch.Tensor) else len(tokens)
    print(f"[main] nb tokens encodes: {n_tokens}")

    if n_tokens <= SEQ_LENGTH:
        raise ValueError(
            f"Pas assez de tokens pour SEQ_LENGTH={SEQ_LENGTH}. tokens={n_tokens}"
        )

    dataset = TextDataset(tokens, SEQ_LENGTH)
    if LOW_RAM_MODE:
        sampler = RandomSampler(dataset, replacement=True, num_samples=len(dataset))
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=DATALOADER_NUM_WORKERS,
            pin_memory=DATALOADER_PIN_MEMORY)
    else:
        dataloader = DataLoader(dataset,
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=DATALOADER_NUM_WORKERS, pin_memory=DATALOADER_PIN_MEMORY)

    steps_per_epoch = len(dataloader)  # ou MAX_STEPS_PER_EPOCH si défini
    total_steps = steps_per_epoch * EPOCHS
    print(f"[main] steps/epoch={steps_per_epoch} | total_steps={total_steps} | warmup_recommandé={int(total_steps*0.02)}")

    #& redéfinission du warmup_steps selon le nombre de steps calculé pour un training complet (2% du total est une règle classique)
    global WARMUP_STEPS
    WARMUP_STEPS = int(total_steps * 0.02)

    print(f"[main] dataset size: {len(dataset)}")
    batch_x, batch_y = next(iter(dataloader))
    print(f"[main] batch input shape: {tuple(batch_x.shape)}")
    print(f"[main] batch target shape: {tuple(batch_y.shape)}")
    return dataloader, len(tokens)


def _train_pipeline(dataloader: DataLoader, tokenizer_model) -> tuple[GPT, dict]:
    """
    Entraîne le modèle GPT sur les données fournies par le dataloader, en utilisant les configurations définies et en sauvegardant les checkpoints et le modèle final

    Args:
        dataloader (DataLoader): Dataloader fournissant les batches d'entraînement
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
        dropout=DROPOUT,
    )

    optimizer = create_adamw_optimizer( model=model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_cfg = TrainConfig(
        epochs=EPOCHS,
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
        warmup_steps=WARMUP_STEPS,
        lr_decay_factor=LR_DECAY_FACTOR,
        use_amp=USE_AMP,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        max_cpu_threads=MAX_CPU_THREADS,
        max_cpu_interop_threads=MAX_CPU_INTEROP_THREADS,
    )

    loss_fn = partial(autoregressive_cross_entropy, ignore_index=LOSS_IGNORE_INDEX,
        label_smoothing=LABEL_SMOOTHING, z_loss_weight=Z_LOSS_WEIGHT)

    history = train_model(model=model, dataloader=dataloader,
        optimizer=optimizer, loss_fn=loss_fn, config=train_cfg)

    if history["epoch_losses"]:
        last_loss = history["epoch_losses"][-1]
        ppl = perplexity_from_loss(last_loss)
        print(f"[main] final loss: {last_loss:.4f} | perplexity: {ppl:.2f}")

    return model, history


def _run_inference(model: GPT, tokenizer_model) -> None:
    """
    Fonction d'inférence qui génère du texte à partir d'un prompt donné, en utilisant le modèle GPT entraîné et le tokenizer pour décoder les tokens générés en texte lisible

    Args:
        model (GPT): Modèle GPT entraîné à utiliser pour l'inférence
        tokenizer_model (_type_): Tokenizer déjà chargé, nécessaire pour décoder les tokens générés en texte lisible
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def _load_model_for_inference_from_artifact(tokenizer_model, final_model_path: str):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _payload = load_model_from_final_state(
        final_model_path=model_path,
        tokenizer=tokenizer_model, device=device)
    print(f"[main] final model loaded for inference: {model_path}")
    return model


def _interactive_reprompt_loop(model: GPT, tokenizer_model) -> None:
    """Permet à l'utilisateur de tester plusieurs prompts en mode interactif sans relancer le pipeline complet."""
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


# *=====================================================================================================================*
# *=====================================================================================================================*
# *=====================================================================================================================*

def main() -> None:
    """
    Fonction principale qui va gérer la pipeline complète du modèle
    """
    print("[main] pipeline start")

    _select_tokenizer_mode_console()
    _select_cache_mode_console()

    if not _validate_files():
        return

    if not _check_dataset_quality(DATASET_JSONL_PATH):
        print("[main] dataset quality check echoue, pipeline stop")
        return

    #* 1) Tokenization/training tokenizer model
    if RUN_TOKENIZER_TRAINING:
        print("[main] tokenization step...")
        td.TokenizeData(mode=TOKENIZER_MODE)

    #* 2) Load tokenizer model
    print("[main] loading tokenizer...")
    try:
        tokenizer_model = td.Tokenizer.load()
    except FileNotFoundError:
        print("[main] tokenizer introuvable. Lance d'abord TokenizeData().")
        return

    #* 3) Build dataloader from encoded corpus
    if INFERENCE_ONLY_FROM_FINAL_MODEL:
        final_model_path = str(Path(FINAL_MODEL_DIR) / FINAL_MODEL_NAME)
        model = _load_model_for_inference_from_artifact(tokenizer_model, final_model_path)
        if model is None:
            print("[main] impossible de charger le modele final, pipeline stop")
            return

        _run_inference(model, tokenizer_model)
        _interactive_reprompt_loop(model, tokenizer_model)
        print("[main] pipeline complete (inference-only mode)")
        return

    dataloader, _num_tokens = _build_dataloader(tokenizer_model)

    #* 4) Train GPT
    model, history = _train_pipeline(dataloader, tokenizer_model)

    #* 5) Checkpoint info
    checkpoint_path = history.get("checkpoint_path")
    final_model_path = history.get("final_model_path")
    print(f"[main] checkpoint path: {checkpoint_path}")
    print(f"[main] final model path: {final_model_path}")

    reloaded_model = _load_model_for_inference_from_artifact(tokenizer_model, final_model_path)
    if reloaded_model is not None:
        model = reloaded_model

    #* 6) Inference test
    _run_inference(model, tokenizer_model)

    #* 7) Interactive reprompt tests
    _interactive_reprompt_loop(model, tokenizer_model)

    print("[main] pipeline complete")


if __name__ == "__main__":
    main()
