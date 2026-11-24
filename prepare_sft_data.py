#!/usr/bin/env python3
"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @Lucas Contreras Hodapp & ChatGPT(LLM - AI)

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
from data.artifacts import atomic_json, sha256_file

DATASET_NAME = "angeluriot/french_instruct"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TOKENIZER_PATH = ROOT_DIR / "tokenizer/token/bpe_model.json"
DEFAULT_OUT_PREFIX = ROOT_DIR / "artifacts/sft/sft"
SPECIAL_TOKENS = ("<|user|>", "<|assistant|>", "<|end|>", "<PAD>")
ALLOWED_SOURCES = ("alpaca", "dolly", "hc3", "human chatgpt comparison corpus", "lima", "less is more for alignment", "aya", "oasst", "open assistant", "ens sup", "doremiti")
BANNED_IN_ANSWER = ("en tant qu'ia", "en tant qu'assistant", "en tant qu'intelligence artificielle", "en tant que modèle de langage", "chatgpt", "openai", "gpt-3", "gpt-4", "je suis un programme informatique")
NOISE_MARKERS = ("url_", "\\pos", "\\displaystyle")
IGNORE_INDEX = -100
LABEL_ALIGNMENT_VERSION = 3
SPLIT_STRATEGY = "prompt_groups_v1"


def prompt_key(example):
    """ Génère une clé unique pour un exemple basé sur le prompt, utilisée pour la déduplication et la séparation train/val.

    Args:
        example (List[tuple(int)]): exemple SFT sous forme de tuple (input_ids, labels).

    Returns:
        tuple: Une clé unique basée sur le prompt, utilisée pour la déduplication et la séparation train/val.
    """
    x, y = example
    start = next(i for i, token in enumerate(y) if token != IGNORE_INDEX)
    return tuple(x[:start + 1])


def split_prompt_groups(examples, val_ratio, seed, group_ids=None, reserved_prompts=(), split_ids=None):
    """Deduplicate pairs and keep every prompt (and declared task) in one split.

    Optional local-data group IDs can also keep paraphrases of one task together.
    Group sizes make the requested validation ratio approximate.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio doit etre compris entre 0 et 1")
    if group_ids is None:
        group_ids = [None] * len(examples)
    if len(group_ids) != len(examples):
        raise ValueError("Un groupe est requis par exemple")
    if split_ids is None:
        split_ids = [None] * len(examples)
    if len(split_ids) != len(examples):
        raise ValueError("Un split est requis par exemple")
    explicit = any(s is not None for s in split_ids)
    if explicit and any(s not in ("train", "dev", "val") for s in split_ids):
        raise ValueError("Les splits locaux doivent tous etre train, dev ou val")
    groups, owners, seen = {}, {}, set()
    assignments = {}
    duplicates = 0
    for example, declared, split in zip(examples, group_ids, split_ids):
        key = prompt_key(example)
        group = ("task", str(declared)) if declared is not None else ("prompt", key)
        if key in owners and owners[key] != group:
            raise ValueError("Une meme question est affectee a plusieurs groupes")
        owners[key] = group
        if explicit:
            split = "val" if split in ("dev", "val") else "train"
            if group in assignments and assignments[group] != split:
                raise ValueError("Un groupe local traverse train et val")
            assignments[group] = split
        pair = (tuple(example[0]), tuple(example[1]))
        if pair in seen:
            duplicates += 1
            continue
        seen.add(pair)
        groups.setdefault(group, []).append(example)
    if len(groups) < 2:
        raise ValueError("Il faut au moins deux groupes de questions distincts pour separer train et val sans fuite")
    reserved = {owners[p] for p in reserved_prompts if p in owners}
    if explicit:
        if any(assignments[g] == "val" for g in reserved):
            raise ValueError("Une question d'identite est presente en validation")
        train = [e for g, values in groups.items() if assignments[g] == "train" for e in values]
        val = [e for g, values in groups.items() if assignments[g] == "val" for e in values]
        if not train or not val:
            raise ValueError("Les splits locaux train et val doivent etre non vides")
        return train, val, {"strategy": SPLIT_STRATEGY, "explicit_splits": True,
            "duplicates_removed": duplicates, "prompt_overlap": 0,
            "train_groups": sum(s == "train" for s in assignments.values()),
            "val_groups": sum(s == "val" for s in assignments.values())}
    keys = [key for key in groups if key not in reserved]
    random.Random(seed).shuffle(keys)
    target = max(1, round(sum(map(len, groups.values())) * val_ratio))
    chosen, count = set(), 0
    for key in keys:
        if len(chosen) + 1 >= len(groups):
            continue  # Always preserve a nonempty training split.
        candidate = count + len(groups[key])
        if candidate <= target or abs(candidate - target) < abs(count - target):
            chosen.add(key)
            count = candidate
    if not chosen:
        if not keys:
            raise ValueError("Tous les groupes sont reserves a l'entrainement")
        chosen.add(min(keys, key=lambda key: len(groups[key])))
    train = [example for key, values in groups.items() if key not in chosen for example in values]
    val = [example for key, values in groups.items() if key in chosen for example in values]
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, {"strategy": SPLIT_STRATEGY, "duplicates_removed": duplicates,
                        "train_groups": len(groups) - len(chosen), "val_groups": len(chosen),
                        "prompt_overlap": 0}


def local_records(path):
    """Read explicit local prompt/answer records without downloading a dataset."""
    with Path(path).open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not all(isinstance(item.get(key), str) and item[key].strip() for key in ("prompt", "answer")):
                raise ValueError(f"Paire prompt/answer invalide: {path}:{line_no}")
            yield {"source": "local", "group": item.get("group"), "split": item.get("split"), "conversation": [
                {"role": "user", "text": item["prompt"]}, {"role": "assistant", "text": item["answer"]}]}


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
    if not all(isinstance(message, dict) for message in conversation):
        return None, "structure_invalide"
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
    # Le logit à la position <|assistant|> doit prédire le premier token de réponse.
    # La dernière position du bloc causal n'a pas de token suivant à prédire.
    labels = [IGNORE_INDEX] * (len(prompt_ids) + 1)
    labels.extend(answer_ids)
    labels.extend([end_id, IGNORE_INDEX])
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
        if not update_path:
            raise ValueError("Tokens SFT manquants avec --no-update-tokenizer: impossible de produire des IDs persistants")
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
            ValueError: Si aucun exemple SFT valide n'est trouvé après le filtrage.
            ValueError: Si le nombre d'exemples SFT valides est inférieur à 2, ce qui est nécessaire pour la séparation train/val.
    """
    

    if args.seq_length <= 0:
        raise ValueError("--seq-length doit etre positif")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio doit etre compris entre 0 et 1")
    if args.max_examples < 0:
        raise ValueError("--max-examples doit etre >= 0 (0 = tous)")
    if args.min_answer_tokens <= 0 or (args.max_answer_tokens is not None and args.max_answer_tokens < args.min_answer_tokens):
        raise ValueError("Bornes de longueur de reponse invalides")
    if args.max_prompt_tokens is not None and args.max_prompt_tokens <= 0:
        raise ValueError("--max-prompt-tokens doit etre positif")
    if args.identity_jsonl and args.identity_repeat < 0:
        raise ValueError("--identity-repeat doit etre >= 0")

    local_path = getattr(args, "local_jsonl", None)
    dataset = list(local_records(local_path)) if local_path else load_dataset(DATASET_NAME, split=args.split)

    if args.list_sources:
        counts = Counter(str(record.get("source")) for record in dataset)
        for source, count in counts.most_common():
            status = "gardee" if source_allowed(source) else "ecartee"
            print(f"[{status}] {count:>7}  {source}")
        return None

    metadata_path = args.out_prefix.with_name(f"{args.out_prefix.name}_meta.json")
    pending_path = metadata_path.with_suffix(".pending")
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path.write_text("Preparation SFT incomplete.\n", encoding="utf-8")
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

        if not local_path and not source_allowed(record.get("source")):
            rejects["source_ecartee"] += 1
            continue

        pair, reason = extract_pair(record, args.context_mode)
        if pair is None:
            rejects[reason or "inconnu"] += 1
            continue
        pairs.append((*pair, record.get("group") if local_path else None, record.get("split") if local_path else None))

    rng.shuffle(pairs)
    target_val = max(1, round(args.max_examples * args.val_ratio))
    target_total = args.max_examples + target_val if args.max_examples else None
    examples: list[tuple[list[int], list[int]]] = []
    group_ids = []
    split_ids = []

    for prompt, answer, group_id, split_id in pairs:
        if target_total is not None and len(examples) >= target_total:
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
        group_ids.append(group_id)
        split_ids.append(split_id)

    if len(examples) < 2:
        raise ValueError("Il faut au moins deux exemples SFT valides pour train et val.")
    identity_examples = []
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

            if args.identity_repeat:
                identity_examples.append(example)
            identity_count += 1
        print(f"Exemples identite ajoutes: {identity_count} x {args.identity_repeat}")

    train, val, split_stats = split_prompt_groups(
        examples, args.val_ratio, args.seed, group_ids,
        reserved_prompts={prompt_key(example) for example in identity_examples}, split_ids=split_ids)
    for example in identity_examples:
        train.extend([example] * args.identity_repeat)
    rng.shuffle(train)

    out_prefix = args.out_prefix
    train_path = out_prefix.with_name(f"{out_prefix.name}_train.npz")
    val_path = out_prefix.with_name(f"{out_prefix.name}_val.npz")
    train_stats = save_split(train_path, train, special_ids[3])
    val_stats = save_split(val_path, val, special_ids[3])

    metadata = {
        "dataset": str(local_path) if local_path else DATASET_NAME,
        "source_sha256": sha256_file(local_path) if local_path else None,
        "split": args.split,
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_vocab_size": len(token_to_id),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "special_tokens": dict(zip(SPECIAL_TOKENS, special_ids)),
        "seq_length": args.seq_length,
        "ignore_index": IGNORE_INDEX,
        "label_alignment_version": LABEL_ALIGNMENT_VERSION,
        "seed": args.seed,
        "context_mode": args.context_mode,
        "train": train_stats,
        "val": val_stats,
        "rejects": dict(rejects),
        "split_groups": split_stats
    }

    metadata_path = out_prefix.with_name(f"{out_prefix.name}_meta.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(metadata_path, metadata)
    pending_path.unlink()

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
    parser.add_argument("--min-answer-tokens", type=int, default=1)
    parser.add_argument("--max-answer-tokens", type=int, default=160)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--context-mode", choices=("merge", "skip", "drop"), default="merge")
    parser.add_argument("--identity-jsonl", type=Path, default=None)
    parser.add_argument("--identity-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--local-jsonl", type=Path, help="Paires locales prompt/answer, groupe de tache optionnel")
    parser.add_argument("--no-update-tokenizer", action="store_true", help="N'enregistre pas les tokens speciaux manquants dans le tokenizer")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
