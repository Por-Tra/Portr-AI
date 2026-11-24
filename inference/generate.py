# -*- coding: utf-8 -*-

"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @Lucas Contreras Hodapp

File name : generate.py


description du script:
    Script d'inférence pour générer du texte à partir d'un modèle GPT entraîné.

    Fonctionnalités :
        - Chargement d'un checkpoint ou artefact final de modèle.
        - Génération de texte à partir d'un prompt avec contrôle de la température
          et du top-k sampling.
        - Décodage des ids de tokens générés en texte lisible, en gérant les
          subtilités du BPE (tokens finissant par </w>, préfixe ▁, etc.).
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
from tokenizer.tokenizer import SPACE_TOKEN


#! Tokens spéciaux à ignorer au décodage
_SPECIAL_TOKENS = {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "<|user|>", "<|assistant|>", "<|end|>"}


def load_model_checkpoint(model: torch.nn.Module,checkpoint_path: str | Path,map_location: str = "cpu") -> dict:
    """Charge les poids depuis un checkpoint d'entraînement.

    Args:
        model (torch.nn.Module): modèle sauvegarder
        checkpoint_path (str | Path): chemin du checkpoint 
        map_location (str, optional): Emplacement de la mémoire pour le chargement des poids. Defaults to "cpu".

    Returns:
        dict: Dictionnaire contenant les informations du checkpoint (état du modèle, hyperparamètres, etc.).
    """
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    return payload


def load_model_from_final_state( final_model_path, tokenizer, device = "cpu"):
    """Reconstruit et charge un modèle GPT depuis l'artefact final d'inférence.

    Args:
        final_model_path (str | Path): Chemin vers l'artefact final (.pt).
        tokenizer (Tokenizer): Tokenizer déjà chargé.
        device (str): Device cible pour l'inférence.

    Returns:
        tuple[torch.nn.Module, dict]: model en eval mode, payload brut.
    """
    payload = torch.load(final_model_path, map_location="cpu", weights_only=True)
    hparams = payload.get("model_hparams") or {}

    #& Reconstruire le modèle avec les hyperparamètres sauvegardés
    vocab_size = int(hparams.get("vocab_size") or len(tokenizer.get_token_to_id()))
    seq_length = int(hparams.get("seq_length") or 128)
    embedding_dim = int(hparams.get("embedding_dim") or 256)
    num_heads = int(hparams.get("num_heads") or 8)
    num_blocks = int(hparams.get("num_blocks") or 4)
    dropout = float(hparams.get("dropout", 0.1))
    expected_fingerprint = payload.get("tokenizer_fingerprint")

    if expected_fingerprint and tokenizer.fingerprint(vocab_size=vocab_size) != expected_fingerprint:
        raise RuntimeError("Tokenizer incompatible avec les IDs et regles BPE du modele sauvegarde")

    model = GPT(
        vocab_size=vocab_size,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        dropout=dropout
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def decode_token_ids(token_ids, tokenizer):
    """
    Décode une liste d'ids de tokens en texte lisible.

    Logique BPE :
    - Le début de mot est marqué par _ (ou ▁ dans les anciens modèles) -> espace.
    - Les autres tokens sont concaténés directement.
    - Les tokens spéciaux  sont ignorés.
    - Compatibilité anciens modèles : un token finissant par </w> marque la
      fin d'un mot -> marqueur retiré et espace ajouté.

    Args:
        token_ids (list[int]): Liste d'entiers (ids de tokens).
        tokenizer (Tokenizer): Instance Tokenizer avec get_id_to_token().

    Returns:
        str: Texte décodé, nettoyé des espaces superflus.
    """
    id_to_token = tokenizer.get_id_to_token()
    end_of_word = getattr(tokenizer, "end_of_word", "")
    pieces: list[str] = []

    for idx in token_ids:
        tok = id_to_token.get(int(idx), "<UNK>")

        #& Ignorer les tokens spéciaux
        if tok in _SPECIAL_TOKENS:
            continue

        if end_of_word and tok.endswith(end_of_word):
            tok = tok[: -len(end_of_word)] + " "

        # Le tokenizer actuel encode l'espace comme préfixe `_`.
        # Convertir le marqueur token par token évite de modifier les `_` internes aux tokens.
        if tok.startswith(SPACE_TOKEN):
            tok = " " + tok[len(SPACE_TOKEN):]
        elif tok.startswith("▁"):
            tok = " " + tok[1:]
        pieces.append(tok)

    text = "".join(pieces)
    #& Supprimer les espaces multiples et les espaces en bord de chaîne
    return " ".join(text.split())


def build_sft_prompt_ids(prompt, tokenizer):
    """Construit les IDs du template utilisé par prepare_sft_data.py

    Args:
        prompt (str): Texte de l'instruction à fournir au modèle.
        tokenizer (Tokenizer): Instance de Tokenizer.

    Raises:
        ValueError: Si le tokenizer ne contient pas les tokens nécessaires pour le template SFT.

    Returns:
        list[int]: Liste des IDs des tokens du template SFT.
    """
    token_to_id = tokenizer.get_token_to_id()
    required_tokens = ("<|user|>", "<|assistant|>", "<|end|>")
    missing = [token for token in required_tokens if token not in token_to_id]

    if missing: raise ValueError(f"Tokens SFT absents du tokenizer: {', '.join(missing)}")

    return [int(token_to_id["<|user|>"]), *tokenizer.encode(prompt), int(token_to_id["<|assistant|>"])]


def generate_from_prompt(model, tokenizer, prompt, max_new_tokens = 40,temperature = 1.0, top_k = 20, device = "cpu", use_sft_template = True):
    """Encode un prompt, génère une réponse, puis la décode.

    Par défaut, le prompt suit le template SFT <|user|> ... <|assistant|>
    et la génération s'arrête sur <|end|>. Pour une continuation de corpus
    pré-entraîné, passer use_sft_template=False.

    Args:
        model (torch.nn.Module): Modèle GPT en mode évaluation.
        tokenizer (Tokenizer): Instance de Tokenizer.
        prompt (str): Texte de départ pour la génération.
        max_new_tokens (int, optional): Nombre de nouveaux tokens à générer. Defaults to 40.
        temperature (float, optional): Température d'échantillonnage (< 1 = plus conservateur). Defaults to 1.0.
        top_k (Optional[int], optional): Si défini, échantillonne parmi les top-k tokens. Defaults to 20.
        device (str, optional): Device cible ("cpu", "cuda", ...). Defaults to "cpu".
        use_sft_template (bool, optional): Utilise le template instruction/réponse SFT. Defaults to True.

    Raises:
        ValueError: Si le prompt est vide.
        ValueError: Si max_new_tokens n'est pas un entier positif.

    Returns:
        Tuple[list[int], str]: IDs complets (prompt inclus), et texte de la continuation seule.
    """
    token_to_id = tokenizer.get_token_to_id()
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 0:
        raise ValueError("max_new_tokens doit etre un entier >= 0")
    
    if use_sft_template:
        prompt_ids = build_sft_prompt_ids(prompt, tokenizer)
        stop_token_id = token_to_id["<|end|>"]
    else:
        prompt_ids = tokenizer.encode(prompt)
        stop_token_id = token_to_id.get("<EOS>")

    if not prompt_ids: 
        prompt_ids = [token_to_id.get("<BOS>", 0)]

    vocab_size = int(getattr(model, "vocab_size", 0))

    #& Vérification de la compatibilité du vocabulaire
    checked_ids = prompt_ids + ([stop_token_id] if stop_token_id is not None else [])
    if vocab_size and any(token_id < 0 or token_id >= vocab_size for token_id in checked_ids):
        raise ValueError("Le modèle ne contient pas les tokens du template SFT.\nChargez un modèle redimensionné pour le vocabulaire SFT.")

    input_tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    max_length = len(prompt_ids) + max_new_tokens

    with torch.no_grad():
        out_tokens = model.generate(
            initial_tokens=input_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=stop_token_id
        )

    token_ids = out_tokens[0].detach().cpu().tolist()
    generated_ids = token_ids[len(prompt_ids):]
    text = decode_token_ids(generated_ids, tokenizer)
    return token_ids, text