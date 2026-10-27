#!/usr/bin/env python3
"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : prepare_sft_data.py


description du script:
    Prépare les données pour le fine-tuning supervisé (SFT) à partir du dataset
    "angeluriot/french_instruct" de Hugging Face.

"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tokenizer.tokenizer import Tokenizer
from datasets import load_dataset

DATASET_NAME = "angeluriot/french_instruct"
DEFAULT_TOKENIZER_PATH = Path("tokenizer/token/bpe_model.json")
DEFAULT_OUT_PREFIX = Path("artifacts/sft/sft")
SPECIAL_TOKENS = ("<|user|>", "<|assistant|>", "<|end|>", "<PAD>")
ALLOWED_SOURCES = ("alpaca", "dolly", "hc3", "lima", "aya", "oasst", "open assistant", "ens sup", "doremiti")
BANNED_IN_ANSWER = ("en tant qu'ia", "en tant qu'assistant", "en tant qu'intelligence artificielle", "en tant que modèle de langage", "chatgpt", "openai", "gpt-3", "gpt-4", "je suis un programme informatique")
NOISE_MARKERS = ("url_", "\\pos", "\\displaystyle")
IGNORE_INDEX = -100


def source_allowed(source):
    """ Vérifie si la source d'un enregistrement est autorisée pour le fine-tuning.

    Args:
        source (Any): La source de l'enregistrement à vérifier.

    Returns:
        bool: True si la source est autorisée, False sinon.
    """
    source_text = str(source or "").lower()
    return any(marker in source_text for marker in ALLOWED_SOURCES)


def extract_pair(record, context_mode):
    """ Extrait une paire (prompt, réponse) d'un enregistrement du dataset.

    Args:
        record (dict[str, Any]): Un enregistrement du dataset contenant les champs "conversation" et éventuellement "context".
        context_mode (str): Mode de gestion du contexte. Peut être "merge" pour fusionner le contexte avec le prompt, "skip" pour ignorer les enregistrements avec contexte, ou "drop" pour supprimer le contexte.

    Returns:
        tuple[tuple[str, str] | None, str | None]: Une paire (prompt, réponse) si l'enregistrement est valide, sinon None.
    """
    conversation = record.get("conversation")

    #& Validation de la structure de la conversation
    if not isinstance(conversation, list) or len(conversation) != 2:
        return None, "multi_tour"
    if conversation[0].get("role") != "user" or conversation[1].get("role") != "assistant":
        return None, "roles_inattendus"

    prompt = str(conversation[0].get("text") or "").strip()
    answer = str(conversation[1].get("text") or "").strip()
    context = str(record.get("context") or "").strip()

    if context:
        if context_mode == "skip":
            return None, "avec_contexte"
        if context_mode == "merge":
            prompt = f"{prompt}\n\n{context}"

    #& Validation du contenu du prompt et de la réponse
    if not prompt or not answer:
        return None, "vide"
    if any(marker in answer.lower() for marker in BANNED_IN_ANSWER):
        return None, "phrase_bannie"
    if any(marker in f"{prompt} {answer}".lower() for marker in NOISE_MARKERS):
        return None, "bruit"
    
    return (prompt, answer), None


def build_example(prompt, answer, encode, special_ids, max_len, min_answer_tokens, 
                  max_answer_tokens, max_prompt_tokens):
    """ Construit un exemple d'entraînement à partir d'un prompt et d'une réponse,
    en encodant les tokens et en appliquant les contraintes de longueur.

    Args:
        prompt (str): Le texte du prompt.
        answer (str): La réponse.
        encode (Callable[[str], list[int]]): Fonction pour encoder le texte en ids de tokens.
        special_ids (tuple[int, int, int, int]): Les ids des tokens spéciaux (user, assistant, end, pad).
        max_len (int): La longueur maximale de la séquence d'entrée.
        min_answer_tokens (int): Le nombre minimum de tokens requis pour la réponse.
        max_answer_tokens (int | None): Le nombre maximum de tokens autorisé pour la réponse.
        max_prompt_tokens (int | None): La longueur maximale autorisée pour le prompt.

    Returns:
        tuple[tuple[list[int], list[int]] | None, str | None]: Une paire (input_ids, labels) si l'exemple est valide, sinon None.
    """
    user_id, assistant_id, end_id, pad_id = special_ids
    prompt_ids = encode(prompt)
    answer_ids = encode(answer)

    #& Vérification des contraintes de longueur
    if len(answer_ids) < min_answer_tokens:
        return None, "reponse_trop_courte"
    if max_answer_tokens is not None and len(answer_ids) > max_answer_tokens:
        return None, "reponse_trop_longue"
    if max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens:
        return None, "prompt_trop_long"


    input_ids = [user_id, *prompt_ids, assistant_id, *answer_ids, end_id]
    if len(input_ids) > max_len:
        return None, "trop_long"

    #& Construction des labels avec padding et index d'ignorance
    labels = [IGNORE_INDEX] * (len(prompt_ids) + 2)
    labels.extend(answer_ids)
    labels.append(end_id)
    padding = max_len - len(input_ids)
    input_ids.extend([pad_id] * padding)
    labels.extend([IGNORE_INDEX] * padding)
    return (input_ids, labels), None


def load_identity_examples(path):
    """ Charge les exemples d'identité à partir d'un fichier JSONL.

    Args:
        path (Path): Le chemin vers le fichier JSONL.

    Returns:
        list[tuple[str, str]]: Une liste de paires (prompt, réponse).
    """
    pairs = []
    with path.open(encoding="utf-8") as source:
        for line in source:

            if not line.strip():
                continue

            item = json.loads(line)
            pairs.append((str(item["prompt"]).strip(), str(item["answer"]).strip()))
    
    return pairs


def load_tokenizer(path, update_path = True):
    """ Charge le tokenizer depuis un fichier et complète les tokens spéciaux manquants si nécessaire.

    Args:
        path (Path): Le chemin vers le fichier du tokenizer.
        update_path (bool, optional): Indique si le chemin du tokenizer doit être mis à jour. Defaults to True.

    Raises:
        ValueError: Si des tokens spéciaux sont manquants dans le tokenizer.

    Returns:
        tuple[Tokenizer, tuple[int, int, int, int]]: Le tokenizer chargé et les IDs des tokens spéciaux.
    """
    tokenizer = Tokenizer.load(str(path))
    token_to_id = tokenizer.get_token_to_id()
    missing = [token for token in SPECIAL_TOKENS if token not in token_to_id]

    if missing:
        next_id = max(token_to_id.values(), default=-1) + 1

        for token in missing:
            token_to_id[token] = next_id
            tokenizer.vocab.add(token)
            next_id += 1

        tokenizer.id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        
        if update_path:
            tokenizer.save(str(path))
            print(f"Tokenizer complete avec {len(missing)} tokens speciaux: {path}")

    ids = [int(token_to_id[token]) for token in SPECIAL_TOKENS]

    if len(set(ids)) != len(ids): raise ValueError("Les tokens speciaux doivent avoir des IDs distincts.")
    
    return tokenizer, tuple(ids)


def save_split(path, examples, pad_id):
    """ Enregistre un split d'exemples dans un fichier NPZ compressé.

    Args:
        path (Path): Le chemin vers le fichier de sortie.
        examples (list[tuple[list[int], list[int]]]): Liste des exemples à enregistrer, chaque exemple étant une paire (input_ids, labels).
        pad_id (int): L'ID du token de padding.

    Returns:
        dict[str, int]: Un dictionnaire contenant des statistiques sur le split enregistré, incluant le nombre d'exemples, le nombre de tokens et le nombre de tokens supervisés.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    input_ids = np.asarray([item[0] for item in examples], dtype=np.int32)
    labels = np.asarray([item[1] for item in examples], dtype=np.int32)
    np.savez_compressed(path, input_ids=input_ids, labels=labels)
    
    return {
        "examples": len(examples),
        "tokens": int((input_ids != pad_id).sum()) if len(input_ids) else 0,
        "supervised_tokens": int((labels != IGNORE_INDEX).sum()) if len(labels) else 0
    }


def prepare(args: argparse.Namespace):
    """ Prépare les données pour le fine-tuning supervisé (SFT) à partir du dataset "angeluriot/french_instruct" de Hugging Face.

    Args:
        args (argparse.Namespace): Arguments de la ligne de commande contenant les paramètres de préparation des données.

    Raises:
            ValueError: Si les paramètres fournis sont invalides, tels que des longueurs négatives ou des ratios hors limites.
            ValueError: Si le dataset ne peut pas être chargé.
            ValueError: Si le tokenizer ne peut pas être chargé ou mis à jour.
    """
    

    if args.seq_length <= 0:
        raise ValueError("--seq-length doit etre positif")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio doit etre compris entre 0 et 1")
    if args.max_examples <= 0:
        raise ValueError("--max-examples doit etre positif")

    dataset = load_dataset(DATASET_NAME, split=args.split)

    if args.list_sources:
        counts = Counter(str(source) for source in dataset["source"])
        for source, count in counts.most_common():
            status = "gardee" if source_allowed(source) else "ecartee"
            print(f"[{status}] {count:>7}  {source}")
        return None

    tokenizer, special_ids = load_tokenizer(args.tokenizer_path, update_path=not args.no_update_tokenizer)

    token_to_id = tokenizer.get_token_to_id()
    encode = tokenizer.encode
    print(f"Tokenizer: {args.tokenizer_path} ({len(token_to_id)} tokens)")
    print("Tokens speciaux:", dict(zip(SPECIAL_TOKENS, special_ids)))

    rng = random.Random(args.seed)
    rejects: Counter[str] = Counter()
    pairs: list[tuple[str, str]] = []

    for record in dataset:

        if record.get("code"):
            rejects["code"] += 1
            continue

        if not source_allowed(record.get("source")):
            rejects["source_ecartee"] += 1
            continue

        pair, reason = extract_pair(record, args.context_mode)
        if pair is None:
            rejects[reason or "inconnu"] += 1
            continue
        pairs.append(pair)

    rng.shuffle(pairs)
    target_val = max(1, round(args.max_examples * args.val_ratio))
    target_total = args.max_examples + target_val
    examples: list[tuple[list[int], list[int]]] = []

    for prompt, answer in pairs:
        if len(examples) >= target_total:
            break

        example, reason = build_example(
            prompt,
            answer,
            encode,
            special_ids,
            args.seq_length,
            args.min_answer_tokens,
            args.max_answer_tokens,
            args.max_prompt_tokens
        )

        if example is None:
            rejects[reason or "inconnu"] += 1
            continue

        examples.append(example)

    rng.shuffle(examples)
    val = examples[: min(target_val, len(examples))]
    train = examples[len(val) :]

    if args.identity_jsonl:

        identity_count = 0

        for prompt, answer in load_identity_examples(args.identity_jsonl):

            example, reason = build_example(
                prompt,
                answer,
                encode,
                special_ids,
                args.seq_length,
                args.min_answer_tokens,
                args.max_answer_tokens,
                args.max_prompt_tokens
            )

            if example is None:
                rejects[f"identite_{reason or 'inconnu'}"] += 1
                continue

            train.extend([example] * args.identity_repeat)
            identity_count += 1

        rng.shuffle(train)
        print(f"Exemples identite ajoutes: {identity_count} x {args.identity_repeat}")

    out_prefix = args.out_prefix
    train_path = out_prefix.with_name(f"{out_prefix.name}_train.npz")
    val_path = out_prefix.with_name(f"{out_prefix.name}_val.npz")
    train_stats = save_split(train_path, train, special_ids[3])
    val_stats = save_split(val_path, val, special_ids[3])

    metadata = {
        "dataset": DATASET_NAME,
        "split": args.split,
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_vocab_size": len(token_to_id),
        "special_tokens": dict(zip(SPECIAL_TOKENS, special_ids)),
        "seq_length": args.seq_length,
        "ignore_index": IGNORE_INDEX,
        "seed": args.seed,
        "context_mode": args.context_mode,
        "train": train_stats,
        "val": val_stats,
        "rejects": dict(rejects)
    }

    metadata_path = out_prefix.with_name(f"{out_prefix.name}_meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Train: {train_path} ({len(train)} exemples)")
    print(f"Val:   {val_path} ({len(val)} exemples)")
    print(f"Meta:  {metadata_path}")
    print("Rejets:")
    for reason, count in rejects.most_common():
        print(f"  {reason:<24} {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-sources", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=30_000)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--min-answer-tokens", type=int, default=3)
    parser.add_argument("--max-answer-tokens", type=int, default=160)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--context-mode", choices=("merge", "skip", "drop"), default="merge")
    parser.add_argument("--identity-jsonl", type=Path, default=None)
    parser.add_argument("--identity-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--no-update-tokenizer",
        action="store_true",
        help="N'enregistre pas les tokens speciaux manquants dans le tokenizer"
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
