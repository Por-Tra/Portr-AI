"""
TokenizeData.py

Lance et configure le processus de tokenisation des données textuelles brutes.
Utilise la classe Tokenizer pour entraîner un modèle BPE sur les données
extraites par le crawler, puis sauvegarde ce modèle pour le pipeline GPT.

Optimisations vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| Tout le dataset chargé en RAM   | Générateur streaming : une ligne lue à  |
| (list de strings) avant train() | la fois, jamais tout en mémoire          |
| Logique load/train incorrecte   | Si le modèle existe : on skippe          |
| (on entraîne même si modèle ok) | l'entraînement directement               |
| Recomptage LIMIT_CORPUS dans    | Le générateur délègue la limite à        |
| deux endroits                   | _build_word_freq (déjà implémentée)      |
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

from tokenizer.tokenizer import Tokenizer, DEFAULT_MODEL_PATH

VOCAB_SIZE = 16_000
DATASET_PATH = "data/datasets/dataset.jsonl"

# surcharger le corpus pour qu'il corresponde à la taille du vocab
LIMIT_CORPUS = VOCAB_SIZE * 10  # ~10 occurrences par token en moyenne



def _configure_stdout_for_unicode() -> None:
    """Force UTF-8 output sur les terminaux compatibles."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _stream_texts(path: str) -> Generator[str, None, None]:
    """
    Générateur qui yield les textes du JSONL ligne par ligne.

    Avantage : aucune liste intermédiaire en mémoire — le fichier est lu
    de manière séquentielle. Sur un corpus de plusieurs Go, la différence
    est critique (RAM constante vs RAM ∝ taille du corpus).
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            if text:
                yield text


def TokenizeData(mode: str = "auto") -> Tokenizer:
    """
    Entraîne (ou charge) un tokenizer BPE et le sauvegarde.

    Modes:
        - "auto": charge si existe, sinon entraine
        - "use_existing": charge uniquement (erreur si absent)
        - "retrain": entraine et ecrase le modele existant

    Returns:
        Tokenizer: instance prête à l'emploi.
    """
    _configure_stdout_for_unicode()

    if mode not in {"auto", "use_existing", "retrain"}:
        raise ValueError(f"Mode tokenizer invalide: {mode}")

    # ── Gestion du modèle existant selon mode ────────────────────────────
    model_path = Path(DEFAULT_MODEL_PATH)
    if model_path.exists() and mode in {"auto", "use_existing"}:
        print(f"[TokenizeData] Modèle existant chargé depuis {model_path}")
        return Tokenizer.load(str(model_path))

    if (not model_path.exists()) and mode == "use_existing":
        raise FileNotFoundError(
            f"Tokenizer introuvable en mode use_existing: {model_path}"
        )

    # ── Entraînement depuis le corpus streamé ─────────────────────────────
    if mode == "retrain" and model_path.exists():
        print("[TokenizeData] Mode retrain actif — ré-entraînement forcé...")
    else:
        print("[TokenizeData] Aucun modèle trouvé — entraînement en cours...")

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET_PATH}")

    tok = Tokenizer()
    tok.train(
        _stream_texts(DATASET_PATH),   # <- générateur, pas une liste
        target_vocab_size=VOCAB_SIZE,
        min_pair_freq=2,
        verbose=True,
        corpus_limit=LIMIT_CORPUS,
    )

    tok.save()
    print("[TokenizeData] Modèle sauvegardé.")
    return tok