"""
generate.py

Inference helpers for GPT generation.

Corrections vs version précédente :
──────────────────────────────────────────────────────────────────────────────
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| decode_token_ids : suppression  | Les tokens terminant par </w> injectent  |
| brute de </w> -> mots concaténés | un espace après eux avant concaténation  |
| sans espaces                    |                                          |
| Tokens spéciaux inclus dans le  | Tokens spéciaux filtrés au décodage      |
| texte généré (<BOS>, <PAD>...)  |                                          |
| sys.path.insert() sur MODEL_DIR | Inutile : gpt_model est importé depuis  |
| (chemin déjà couvert par ROOT)  | le package model/ normalement            |
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.gpt_model import GPT


# Tokens spéciaux à ignorer au décodage
_SPECIAL_TOKENS = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}


def load_model_checkpoint(model: torch.nn.Module,checkpoint_path: str | Path,map_location: str = "cpu") -> dict:
    """Charge les poids depuis un checkpoint d'entraînement."""
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def load_model_from_final_state( final_model_path: str | Path, tokenizer, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Reconstruit et charge un modèle GPT depuis l'artefact final d'inférence.

    Args:
        final_model_path : Chemin vers l'artefact final (.pt).
        tokenizer        : Tokenizer déjà chargé.
        device           : Device cible pour l'inférence.

    Returns:
        Tuple (model en eval mode, payload brut).
    """
    payload = torch.load(final_model_path, map_location="cpu", weights_only=True)
    hparams = payload.get("model_hparams") or {}

    vocab_size    = int(hparams.get("vocab_size")    or len(tokenizer.get_token_to_id()))
    seq_length    = int(hparams.get("seq_length")    or 128)
    embedding_dim = int(hparams.get("embedding_dim") or 256)
    num_heads     = int(hparams.get("num_heads")     or 8)
    num_blocks    = int(hparams.get("num_blocks")    or 4)
    dropout       = float(hparams.get("dropout")     or 0.1)

    model = GPT(
        vocab_size=vocab_size,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def decode_token_ids(token_ids: list[int], tokenizer) -> str:
    """
    Décode une liste d'ids de tokens en texte lisible.

    Logique BPE :
    - Un token qui se termine par </w> marque la fin d'un mot.
      On retire le marqueur et on ajoute un espace APRÈS ce token,
      ce qui sépare correctement les mots dans le texte reconstitué.
    - Un token sans </w> est un sous-mot intermédiaire : on le concatène
      directement au suivant, sans espace.
    - Les tokens spéciaux (<PAD>, <BOS>, <EOS>, <UNK>) sont ignorés.
    - Le préfixe ▁ (SentencePiece style) est converti en espace.

    Args:
        token_ids : Liste d'entiers (ids de tokens).
        tokenizer : Instance Tokenizer avec get_id_to_token().

    Returns:
        Texte décodé, nettoyé des espaces superflus.
    """
    id_to_token = tokenizer.get_id_to_token()
    end_of_word = getattr(tokenizer, "end_of_word", "</w>")
    pieces: list[str] = []

    for idx in token_ids:
        tok = id_to_token.get(int(idx), "<UNK>")

        # Ignorer les tokens spéciaux
        if tok in _SPECIAL_TOKENS:
            continue

        if tok.endswith(end_of_word):
            # Fin de mot : retirer le marqueur, ajouter un espace séparateur
            pieces.append(tok[: -len(end_of_word)] + " ")
        else:
            # Sous-mot intermédiaire : concaténation directe
            pieces.append(tok)

    text = "".join(pieces)
    # Normaliser le préfixe ▁ (utilisé par certains tokenizers comme marqueur d'espace)
    text = text.replace("▁", " ")
    # Supprimer les espaces multiples et les espaces en bord de chaîne
    return " ".join(text.split())


def generate_from_prompt(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    top_k: Optional[int] = 20,
    device: str = "cpu",
) -> tuple[list[int], str]:
    """Encode un prompt, génère une continuation, puis décode.

    Args:
        model          : Modèle GPT en eval mode.
        tokenizer      : Tokenizer chargé.
        prompt         : Texte de départ.
        max_new_tokens : Nombre de nouveaux tokens à générer.
        temperature    : Température d'échantillonnage (< 1 = plus conservateur).
        top_k          : Si défini, échantillonne parmi les top-k tokens.
        device         : Device cible ("cpu", "cuda", ...).

    Returns:
        Tuple (liste d'ids générés, texte décodé).
    """
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.get_token_to_id().get("<BOS>", 0)]

    input_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    max_length = len(prompt_ids) + max_new_tokens

    with torch.no_grad():
        out_tokens = model.generate(
            initial_tokens=input_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

    token_ids = out_tokens[0].detach().cpu().tolist()
    text = decode_token_ids(token_ids, tokenizer)
    return token_ids, text