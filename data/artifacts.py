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

File name : dataset.py

description du script:

    Fichier définissant la classe TextDataset, un wrapper PyTorch pour les données textuelles tokenisées.


"""
from pathlib import Path
import hashlib
import json
import os
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    """ 
    Écrit atomiquement un dictionnaire JSON dans un fichier.

    Args:
        path (Path): Le chemin vers le fichier JSON.
        value (dict): La valeur à écrire dans le fichier.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def file_identity(path):
    """ Détermine l'identité d'un fichier en termes de chemin, taille et date de modification.


    Args:
        path (Path): Le chemin vers le fichier.

    Returns:
        dict: Un dictionnaire contenant l'identité du fichier.
    """
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256_file(path):
    """ Calcule le hash SHA256 d'un fichier.

    Args:
        path (Path): Le chemin vers le fichier.

    Returns:
        str: Le hash SHA256 du fichier.
    """
    digest = hashlib.sha256()

    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_token_cache(manifest, tokenizer, *, allow_empty=False):
    """ Charge un cache de tokens à partir d'un manifeste JSON et d'un tokenizer donné.

    Args:
        manifest (path): _description_
        tokenizer (Tokenizer): _description_
        allow_empty (bool, optional): _description_. Defaults to False.

    Raises:
        RuntimeError: Erreur de cache
        ValueError: problème de tokens
        ValueError: manifest invalide
        ValueError:  tokenizer invalide
        RuntimeError: Cache invalide

    Returns:
        torch.Tensor: Les tokens chargés depuis le cache.
    """
    manifest = Path(manifest).resolve()
    if (manifest.parent / ".preparation_in_progress").exists():
        raise RuntimeError("Preparation de cache incomplete")
    
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        count = payload["token_count"]

        if payload.get("dtype") != "int32" or type(count) is not int or count < 0 or (count == 0 and not allow_empty):
            raise ValueError("dtype ou token_count invalide")
        
        binary = Path(payload["bin_path"])
        if not binary.is_absolute():
            candidates = (manifest.parent / binary, ROOT_DIR / binary, manifest.parent / binary.name)
            binary = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])

        if not binary.is_file() or binary.stat().st_size != count * 4:
            raise ValueError("binaire absent ou taille incoherente")
        
        if payload.get("tokenizer_fingerprint"):
            size = payload["tokenizer_vocab_size"]

            if tokenizer.fingerprint(vocab_size=size) != payload["tokenizer_fingerprint"]:
                raise ValueError("Tokenizer incompatible avec les tokens encodes")
            
        return torch.from_file(str(binary), shared=False, size=count, dtype=torch.int32)
    
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        raise RuntimeError(f"Cache invalide: {manifest}: {exc}") from exc
