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
    
    Ce script prépare les données pour l'entraînement d'un modèle de langage. Il effectue les étapes suivantes :
    1. Télécharge et charge le dataset "OpenLLM-France/Luciole-Training-Dataset" avec la configuration "Fineweb2-HQ".
    2. Nettoie le texte en supprimant les caractères de contrôle et en normalisant les espaces.
    3. Génère un échantillon de texte pour entraîner le tokenizer.
    4. Entraîne un tokenizer BPE sur l'échantillon de texte et le sauvegarde.
    5. Divise les documents en ensembles d'entraînement et de validation selon un pourcentage spécifié.
    6. Encode les documents en tokens à l'aide du tokenizer entraîné et sauvegarde les résultats dans des fichiers binaires.

MEMO : python prepare_data.py --split fr --token-budget 500000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from array import array
from pathlib import Path
from typing import Iterable, Iterator

from datasets import get_dataset_split_names, load_dataset
from tqdm import tqdm

from tokenizer.tokenizer import Tokenizer


DATASET_NAME = "OpenLLM-France/Luciole-Training-Dataset"
DATASET_CONFIG = "Fineweb2-HQ"
DEFAULT_SPLIT = "fr"
DEFAULT_TOKENIZER_SAMPLE_BYTES = 300 * 1024 * 1024
DEFAULT_TOKEN_BUDGET = 500_000_000
DEFAULT_VAL_PERCENT = 1
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path("data/datasets")
DEFAULT_CACHE_DIR = Path("artifacts/cache")
DEFAULT_TOKENIZER_PATH = Path("tokenizer/token/bpe_model.json")
EOS_TOKEN = "<EOS>"

CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(value):
    if not isinstance(value, str):
        return ""
    text = CONTROL_CHARS.sub(" ", value).replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def document_id(doc, text):
    value = doc.get("id")
    if value is not None and str(value).strip():
        return str(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_validation(doc_id, percent):
    bucket = int.from_bytes(hashlib.sha256(doc_id.encode("utf-8")).digest()[:8], "big") % 100
    return bucket < percent


def stream_documents(split, seed):
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=split,
        streaming=True
    ).shuffle(seed=seed, buffer_size=10_000)

    for doc in dataset:
        text = clean_text(doc.get("text"))
        if len(text) < 40:
            continue
        yield document_id(doc, text), text


def write_tokenizer_sample(split, seed, output_path, sample_bytes, val_percent):
    """ Ecrit un échantillon de texte pour entraîner le tokenizer, en respectant la taille maximale spécifiée.

    Args:
        split (str): nom du split du dataset à utiliser (ex: "fr")
        seed (int): graine pour le mélange aléatoire des documents
        output_path (Path): chemin du fichier de sortie pour l'échantillon JSONL
        sample_bytes (int): taille maximale de l'échantillon en octets
        val_percent (int): pourcentage de documents à utiliser pour la validation

    Returns:
        int: taille totale de l'échantillon écrit en octets
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8") as output:
        for doc_id, text in stream_documents(split, seed):
            if is_validation(doc_id, val_percent):
                continue
            output.write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
            total += len(text.encode("utf-8"))
            if total >= sample_bytes:
                break
    return total


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text", "")
            if text:
                yield str(item.get("id", "")), text


def write_jsonl_and_count(split, seed, output_dir, cache_dir, token_budget, tokenizer, val_percent):
    """ Ecrit les documents dans des fichiers JSONL séparés pour l'entraînement et la validation, en respectant le budget de tokens.

    Args:
        split (str): nom du split
        seed (int): graine pour génération aléatoire
        output_dir (Path): chemin de sortie
        token_budget (int): nombre de token de génération
        tokenizer (Tokenizer): Instance de génération de tokens
        val_percent (int): pourcentage de documents à utiliser pour la validation

    Returns:
        tuple[int, int, int]: nombre total de documents, nombre de tokens pour l'entraînement, nombre de tokens pour la validation
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / "val.jsonl").open("w", encoding="utf-8")
    }
    binary_handles = {
        split_name: (cache_dir / f"{split_name}_tokens.bin").open("wb")
        for split_name in ("train", "val")
    }
    buffers = {"train": array("i"), "val": array("i")}

    counts = {"train": 0, "val": 0}
    documents = 0
    try:
        with tqdm(total=token_budget, desc="Collecting and encoding", unit="tok") as progress:
            for doc_id, text in stream_documents(split, seed):
                encoded = tokenizer.encode(text)
                encoded.append(tokenizer.get_token_to_id()[EOS_TOKEN])
                encoded_count = len(encoded)
                if counts["train"] + counts["val"] >= token_budget:
                    break
                target = "val" if is_validation(doc_id, val_percent) else "train"
                handles[target].write(json.dumps({"id": doc_id, "text": text}, ensure_ascii=False) + "\n")
                buffers[target].extend(encoded)
                counts[target] += encoded_count
                documents += 1
                progress.update(encoded_count)
                if len(buffers[target]) >= 1_000_000:
                    buffers[target].tofile(binary_handles[target])
                    buffers[target] = array("i")
    finally:
        for handle in handles.values():
            handle.close()
        for split_name, handle in binary_handles.items():
            if buffers[split_name]:
                buffers[split_name].tofile(handle)
            handle.close()
            (cache_dir / f"{split_name}_tokens.json").write_text(
                json.dumps(
                    {
                        "split": split_name,
                        "token_count": counts[split_name],
                        "dtype": "int32",
                        "bin_path": str(cache_dir / f"{split_name}_tokens.bin"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    return documents, counts["train"], counts["val"]


def encode_jsonl(path, cache_dir, tokenizer, split):
    """ Encode les documents d'un fichier JSONL en tokens et les écrit dans un fichier binaire, en respectant le split spécifié.

    Args:
        path (Path): chemin du fichier JSONL à encoder
        cache_dir (Path): répertoire de cache pour les fichiers binaires et manifestes
        tokenizer (Tokenizer): Instance de génération de tokens
        split (str): nom du split (ex: "train" ou "val")

    Returns:
        int: nombre total de tokens encodés et écrits dans le fichier binaire
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    bin_path = cache_dir / f"{split}_tokens.bin"
    manifest_path = cache_dir / f"{split}_tokens.json"
    token_count = 0
    buffer = array("i")
    eos_id = tokenizer.get_token_to_id()[EOS_TOKEN]

    with bin_path.open("wb") as output:
        for _, text in tqdm(iter_jsonl(path), desc=f"Encoding {split}"):
            buffer.extend(tokenizer.encode(text))
            buffer.append(eos_id)
            if len(buffer) >= 1_000_000:
                buffer.tofile(output)
                token_count += len(buffer)
                buffer = array("i")
        if buffer:
            buffer.tofile(output)
            token_count += len(buffer)

    manifest_path.write_text(
        json.dumps(
            {"split": split, "token_count": token_count, "dtype": "int32", "bin_path": str(bin_path)},
            indent=2,
        ),
        encoding="utf-8"
    )
    return token_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--tokenizer-sample-bytes", type=int, default=DEFAULT_TOKENIZER_SAMPLE_BYTES)
    parser.add_argument("--val-percent", type=int, default=DEFAULT_VAL_PERCENT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vocab-size", type=int, default=16_000)
    return parser.parse_args()


def prepare_dataset(
    split=DEFAULT_SPLIT,
    token_budget=DEFAULT_TOKEN_BUDGET,
    tokenizer_sample_bytes=DEFAULT_TOKENIZER_SAMPLE_BYTES,
    val_percent=DEFAULT_VAL_PERCENT,
    seed=DEFAULT_SEED,
    output_dir=DEFAULT_OUTPUT_DIR,
    cache_dir=DEFAULT_CACHE_DIR,
    tokenizer_path=DEFAULT_TOKENIZER_PATH,
    vocab_size=16_000
):
    if not 0 <= val_percent < 100:
        raise ValueError("--val-percent doit etre compris entre 0 et 99")

    print(f"[prepare:1/6] verification des splits: {DATASET_NAME} / {DATASET_CONFIG}")
    available_splits = get_dataset_split_names(DATASET_NAME, DATASET_CONFIG)
    print(f"Splits disponibles pour {DATASET_CONFIG}: {available_splits}")
    if split not in available_splits:
        raise ValueError(f"Split '{split}' absent. Choisir parmi: {available_splits}")

    sample_path = output_dir / "tokenizer_sample.jsonl"
    print(f"[prepare:2/6] echantillonnage du split '{split}' ({tokenizer_sample_bytes / 1024**2:.0f} MiB maximum)")
    sample_bytes = write_tokenizer_sample(split, seed, sample_path, tokenizer_sample_bytes, val_percent)
    print(f"[prepare:3/6] echantillon termine: {sample_bytes / 1024**2:.1f} MiB dans {sample_path}")
    tokenizer = Tokenizer()
    print(f"[prepare:4/6] entrainement BPE (vocabulaire cible: {vocab_size})")
    tokenizer.train(
        (text for _, text in iter_jsonl(sample_path)),
        target_vocab_size=vocab_size,
        min_pair_freq=2,
        verbose=True
    )
    tokenizer.save(str(tokenizer_path))
    print(f"[prepare:5/6] tokenizer sauvegarde: {tokenizer_path}")

    print(f"[prepare:6/6] collecte + encodage jusqu'a {token_budget:,} tokens")
    documents, train_budget, val_budget = write_jsonl_and_count(
        split, seed, output_dir, cache_dir, token_budget, tokenizer, val_percent
    )
    train_tokens = train_budget
    val_tokens = val_budget
    print(f"Prepared {documents} documents ({sample_bytes / 1024**2:.1f} MiB tokenizer sample); "
        f"train={train_tokens} tokens, val={val_tokens} tokens "
        f"(estimated {train_budget}/{val_budget})."
    )


def main():
    args = parse_args()
    prepare_dataset(
        split=args.split,
        token_budget=args.token_budget,
        tokenizer_sample_bytes=args.tokenizer_sample_bytes,
        val_percent=args.val_percent,
        seed=args.seed,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()