"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : repair_sft.py


Build verified French instruction exercises and run isolated SFT repairs.

No base training, no changes to original datasets or the selected model. A fixed
test set is generated before training and kept separate from model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main
import prepare_sft_data as prep
from data.artifacts import atomic_json, sha256_file
from data.dataset import ResumableRandomSampler
from data.sft_dataset import SFTDataset
from inference.generate import build_sft_prompt_ids, decode_token_ids, load_model_from_final_state
from tokenizer.tokenizer import Tokenizer
from training.sft_train import SFTConfig, train_sft_model

SEED = 73921
OUT = ROOT / "artifacts/sft/repairs/20261116"


def dump_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def protect():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "protected.json"
    if not path.exists():
        files = [main._base_model_path(), Path(main.CHECKPOINT_DIR) / main.CHECKPOINT_NAME,
                 Path(main.TOKENIZER_MODEL_PATH), main._sft_model_path(),
                 main.SFT_TRAIN_PATH, main.SFT_VAL_PATH, main.SFT_META_PATH]
        atomic_json(path, {"files": {str(p.resolve()): sha256_file(p) for p in files},
                           "original_sft": str(main._sft_model_path().resolve()),
                           "active_selection": main.SFT_ACTIVE_RUN_PATH.read_text(encoding="utf-8")})
    result = json.loads(path.read_text(encoding="utf-8"))
    for filename, digest in result["files"].items():
        if sha256_file(filename) != digest:
            raise RuntimeError(f"Protected input changed: {filename}")
    return result


def curriculum():
    """Answers come from arithmetic/string rules and a small explicit fact bank.

    A semantic instance, across its paraphrases, belongs to only one split.
    Facts are scored separately as acquired knowledge, with unseen phrasings.
    """
    rng = random.Random(SEED)
    pool, test = [], []

    def add(family, key, prompts, answer, knowledge=False):
        group = family + ":" + str(key)
        is_test = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 7 == 0
        for i, prompt in enumerate(prompts):
            record = {"prompt": prompt, "answer": str(answer), "family": family, "group": group}
            if knowledge:
                # Knowledge is taught, then tested with an unseen formulation.
                record["group"] = group + ":phrasing:" + str(i)
                (test if i == len(prompts) - 1 else pool).append(record)
            else:
                (test if is_test else pool).append(record)

    # Held-out words include unseen strings; no training answer is copied into a
    # test example under another template.
    syllables = ["ma", "lo", "pi", "ta", "ru", "ne", "fi", "sa", "bo", "di", "la", "po"]
    words = {"".join(rng.choices(syllables, k=rng.choice((2, 3)))) for _ in range(1200)}
    words.update("arbre bateau maison chien chat rouge bleu vert pomme soleil lune livre table porte fleur jardin".split())
    for word in sorted(words):
        add("copie", word, [f"Recopie seulement le mot suivant : {word}",
            f"Écris exactement « {word} », sans ajouter de commentaire.",
            f"Réponds uniquement avec ce mot : {word}", f"Mot à copier : {word}. Réponds par ce mot seul."], word)
    pairs = [(a, b) for a in range(21) for b in range(21)]
    for a, b in pairs:
        add("addition", (a, b), [f"Calcule {a} + {b}. Réponds uniquement par le nombre.",
            f"Combien font {a} plus {b} ? Donne seulement le résultat.",
            f"Réponds par un nombre : {a} + {b} = ?"], a+b)
        if a >= b:
            add("soustraction", (a, b), [f"Calcule {a} - {b}. Réponds uniquement par le nombre.",
                f"Combien font {a} moins {b} ? Donne seulement le résultat.",
                f"Réponds par un nombre : {a} - {b} = ?"], a-b)
        add("comparaison", (a, b), [f"Entre {a} et {b}, quel nombre est le plus grand ? Réponds par le nombre seul.",
            f"Donne uniquement le maximum de ces nombres : {a}, {b}."], max(a, b))
    names = "Léa Paul Emma Hugo Alice Lucas Julie Marc Nina Adam Émile Inès Léo Clara Louis Sarah Tom Zoé Max Rose".split()
    cities = "Paris Lyon Lille Nice Nantes Rennes Rouen Tours Brest Metz Dijon Caen Rome Madrid Berlin Lisbonne Londres Dublin Vienne Oslo".split()
    for person in names:
        for city in cities:
            add("lecture", (person, city), [f"Texte : {person} habite à {city}. Dans quelle ville habite {person} ? Réponds uniquement par la ville.",
                f"{person} vit dans la ville de {city}. Donne seulement le nom de cette ville.",
                f"Lis : « {person} réside à {city}. » Où réside {person} ? Réponds par la ville seule."], city)
    objects = "ballon livre crayon vélo sac jeton ruban cahier manteau cube bateau stylo vase bol camion pull".split()
    colors = "rouge bleu vert jaune noir blanc orange rose violet gris".split()
    for obj in objects:
        for color in colors:
            add("couleur", (obj, color), [f"Le {obj} est {color}. Quelle est sa couleur ? Réponds par la couleur seule.",
                f"Un {obj} de couleur {color} se trouve sur la table. Donne uniquement sa couleur.",
                f"Couleur du {obj} : {color}. Réponds uniquement par cette couleur."], color)
    items = "pomme poire livre crayon fleur étoile cube ballon arbre chaise".split()
    lists = set()
    for _ in range(700):
        lists.add(tuple(rng.sample(items, rng.randint(1, 6))))
    for values in sorted(lists):
        text = ", ".join(values)
        add("comptage", values, [f"Liste : {text}. Combien y a-t-il d'éléments ? Réponds uniquement par le nombre.",
            f"Compte les éléments séparés par des virgules : {text}. Donne seulement leur nombre."], len(values))
    facts = [
        ("Quelle est la capitale de la France ?", "Paris"), ("Quelle est la capitale de l'Italie ?", "Rome"),
        ("Quelle est la capitale de l'Espagne ?", "Madrid"), ("Quelle est la capitale de l'Allemagne ?", "Berlin"),
        ("Quelle est la capitale du Portugal ?", "Lisbonne"), ("Quelle est la capitale de la Belgique ?", "Bruxelles"),
        ("Combien de jours compte une semaine ?", "7"), ("Combien d'heures compte une journée ?", "24"),
        ("Combien de minutes compte une heure ?", "60"), ("Combien de secondes compte une minute ?", "60"),
        ("Combien de mois compte une année ?", "12"), ("Combien de côtés possède un triangle ?", "3"),
        ("Combien de côtés possède un carré ?", "4"), ("Quel animal miaule ?", "chat"),
        ("Quel animal aboie ?", "chien"), ("Traduis en français le mot anglais dog.", "chien"),
        ("Traduis en français le mot anglais cat.", "chat"), ("Traduis en français le mot anglais house.", "maison"),
        ("Traduis en français le mot anglais red.", "rouge"), ("Traduis en français le mot anglais blue.", "bleu"),
        ("Comment t'appelles-tu ?", "Portr-AI"), ("Quel est ton nom ?", "Portr-AI"),
        ("Dis bonjour en français.", "Bonjour"), ("Dis merci en français.", "Merci"),
        ("Explique simplement ce qu'est un arbre.", "Un arbre est une plante qui possède un tronc, des branches et des feuilles."),
        ("Explique simplement ce qu'est un triangle.", "Un triangle est une figure géométrique qui possède trois côtés."),
        ("Comment préparer une omelette ?", "Bats des œufs avec une pincée de sel. Verse-les dans une poêle chaude et fais-les cuire."),
    ]
    wrappers = ["{q} Réponds sans commentaire supplémentaire.", "Question : {q} Donne seulement la réponse.",
                "Réponds brièvement. {q}", "Réponds directement à cette question : {q}",
                "{q} Donne une réponse courte.", "{q}",
                "S'il te plaît, réponds à la question suivante sans rien ajouter : {q}"]
    for question, answer in facts:
        add("faits_appris", question, [w.format(q=question) for w in wrappers], answer, knowledge=True)
    # Exact choice is a context/extraction task, not an identity fact.
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        add("choix", letter, [f"Réponds uniquement par la lettre {letter}.",
            f"Écris seulement {letter}, sans autre texte.", f"La lettre choisie est {letter}. Quelle lettre dois-tu écrire ? Réponds seulement par cette lettre."], letter)
    rng.shuffle(pool)
    rng.shuffle(test)
    return pool, test


def balanced(rows, per_family):
    groups = defaultdict(list)
    for row in rows:
        groups[row["family"]].append(row)
    rng = random.Random(SEED)
    result = []
    for key in sorted(groups):
        rng.shuffle(groups[key])
        result.extend(groups[key][:per_family])
    return result


def build():
    pool, test = curriculum()
    dump_lines(OUT / "curriculum.jsonl", pool)
    dump_lines(OUT / "test_locked.jsonl", balanced(test, 30))
    args = argparse.Namespace(seq_length=main.SEQ_LENGTH, val_ratio=.1, max_examples=0, split="local",
        list_sources=False, tokenizer_path=Path(main.TOKENIZER_MODEL_PATH), no_update_tokenizer=True,
        seed=SEED, context_mode="merge", min_answer_tokens=1, max_answer_tokens=160,
        max_prompt_tokens=None, identity_jsonl=None, identity_repeat=0,
        out_prefix=OUT / "sft", local_jsonl=OUT / "curriculum.jsonl")
    prep.prepare(args)
    val = SFTDataset(OUT / "sft_val.npz")
    val_keys = {prep.prompt_key((x.tolist(), y.tolist())) for x, y in zip(val.input_ids, val.labels)}
    tok = Tokenizer.load(main.TOKENIZER_MODEL_PATH)
    val_rows = [r for r in pool if tuple(build_sft_prompt_ids(r["prompt"], tok)) in val_keys]
    dump_lines(OUT / "dev_generation.jsonl", balanced(val_rows, 12))
    train = SFTDataset(OUT / "sft_train.npz")
    train_keys = {prep.prompt_key((x.tolist(), y.tolist())) for x, y in zip(train.input_ids, train.labels)}
    test_keys = {tuple(build_sft_prompt_ids(r["prompt"], tok)) for r in test}
    if train_keys & val_keys or (train_keys | val_keys) & test_keys:
        raise RuntimeError("Curriculum prompt leakage")
    atomic_json(OUT / "curriculum_audit.json", {"train": len(train), "validation": len(val),
        "prompt_overlap": 0, "pool_families": dict(Counter(r["family"] for r in pool)),
        "test_families": dict(Counter(r["family"] for r in read_lines(OUT / "test_locked.jsonl"))),
        "test_sha256": sha256_file(OUT / "test_locked.jsonl"),
        "facts_policy": "Known facts with unseen phrasing are scored separately from held-out rule instances."})


def normalize(text):
    return re.sub(r"\s+", " ", text).strip(" \t\n.!?:;\"'«»").casefold()


@torch.inference_mode()
def score(path, rows, tokenizer, device, destination):
    model, _ = load_model_from_final_state(path, tokenizer, str(device))
    end = tokenizer.get_token_to_id()["<|end|>"]
    answers = []
    started = time.perf_counter()
    # Batch equal-length prompts: no padding and no change of position semantics.
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        ids = build_sft_prompt_ids(row["prompt"], tokenizer)
        buckets[len(ids)].append((i, row, ids))
    for items in buckets.values():
        for start in range(0, len(items), 16):
            batch = items[start:start+16]
            inputs = torch.tensor([r[2] for r in batch], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model.generate(inputs, inputs.shape[1]+48, 0.0, None, end)
            for (i, row, _), generated in zip(batch, tokens[:, inputs.shape[1]:].cpu().tolist()):
                stopped = end in generated
                if stopped:
                    generated = generated[:generated.index(end)+1]
                text = decode_token_ids(generated, tokenizer)
                grams = [tuple(generated[j:j+4]) for j in range(len(generated)-3)]
                answers.append({**row, "index": i, "generated": text,
                    "correct": normalize(text) == normalize(row["answer"]), "stopped": stopped,
                    "repetition": (len(grams)-len(set(grams)))/max(1,len(grams))})
    answers.sort(key=lambda row: row["index"])
    result = {"model": str(path), "examples": len(rows), "seconds": time.perf_counter()-started,
              "accuracy": sum(r["correct"] for r in answers)/len(rows),
              "stop_rate": sum(r["stopped"] for r in answers)/len(rows),
              "repetition": sum(r["repetition"] for r in answers)/len(rows), "families": {}}
    for family in sorted({r["family"] for r in answers}):
        group = [r for r in answers if r["family"] == family]
        result["families"][family] = {"n": len(group), "accuracy": sum(r["correct"] for r in group)/len(group)}
    atomic_json(destination, {"summary": result, "answers": answers})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def mixed_data(count, tok):
    """Replay short original answers, excluding every validation/test prompt.

    The original references are not declared factually verified. Their purpose
    is to retain varied language; the generated curriculum supplies exact tasks.
    """
    directory = OUT / f"mixed{count}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"sft_meta.json").exists():
        meta = json.loads((directory/"sft_meta.json").read_text(encoding="utf-8"))
        for split in ("train", "val"):
            if sha256_file(directory/f"sft_{split}.npz") != meta[f"{split}_sha256"]:
                raise RuntimeError("Mixed dataset changed")
        return directory/"sft_train.npz", directory/"sft_val.npz"
    current_train, current_val = SFTDataset(OUT/"sft_train.npz"), SFTDataset(OUT/"sft_val.npz")
    originals = {Path(p).name:Path(p) for p in protect()["files"]}
    original_train, original_val = SFTDataset(originals["sft_train.npz"]), SFTDataset(originals["sft_val.npz"])
    as_examples = lambda ds: [(x.tolist(), y.tolist()) for x, y in zip(ds.input_ids, ds.labels)]
    validation = as_examples(current_val) + as_examples(original_val)
    blocked = {prep.prompt_key(e) for e in validation}
    blocked.update(tuple(build_sft_prompt_ids(row["prompt"], tok)) for row in read_lines(OUT/"test_locked.jsonl"))
    training = [e for e in as_examples(current_train) if prep.prompt_key(e) not in blocked]
    seen = {prep.prompt_key(e) for e in training}
    candidates = as_examples(original_train)
    random.Random(SEED).shuffle(candidates)
    selected = []
    for x, y in candidates:
        key = prep.prompt_key((x,y))
        target = [t for t in y if t != -100]
        if key in blocked or key in seen or not 4 <= len(target) <= 64:
            continue
        grams = [tuple(target[i:i+4]) for i in range(len(target)-3)]
        if (len(grams)-len(set(grams)))/max(1,len(grams)) > .15:
            continue
        selected.append((x,y))
        seen.add(key)
        if len(selected) == count:
            break
    training += selected
    random.Random(SEED).shuffle(training)
    pad = tok.get_token_to_id()["<PAD>"]
    train_path, val_path = directory/"sft_train.npz", directory/"sft_val.npz"
    train_stats = prep.save_split(train_path, training, pad)
    val_stats = prep.save_split(val_path, validation, pad)
    if {prep.prompt_key(e) for e in training} & {prep.prompt_key(e) for e in validation}:
        raise RuntimeError("Replay split leakage")
    # Source JSONL supports future preparations from the manager without a download.
    foundation_groups = {tuple(build_sft_prompt_ids(row["prompt"],tok)): row["group"]
                         for row in read_lines(OUT/"curriculum.jsonl")}
    local = []
    for x,y in training+validation:
        key = prep.prompt_key((x,y))
        local.append({"prompt":decode_token_ids(key[1:-1],tok),
                      "answer":decode_token_ids([v for v in y if v != -100],tok),
                      "group":foundation_groups.get(key,"replay:"+hashlib.sha256(str(key).encode()).hexdigest())})
    dump_lines(directory/"source.jsonl",local)
    meta = json.loads((OUT/"sft_meta.json").read_text(encoding="utf-8"))
    meta.update(dataset=str(directory/"source.jsonl"), source_sha256=sha256_file(directory/"source.jsonl"),
        dataset_description="verified curriculum plus filtered original replay",train=train_stats, val=val_stats,
        train_sha256=sha256_file(train_path),val_sha256=sha256_file(val_path),
        replay_examples=len(selected), replay_answers_factually_verified=False,
        split_groups={"strategy":prep.SPLIT_STRATEGY,"prompt_overlap":0},
        curriculum_test_sha256=sha256_file(OUT/"test_locked.jsonl"))
    atomic_json(directory/"sft_meta.json",meta)
    return train_path, val_path


def train(name, epochs, lr, device, tok, mix_old=0, init_run=None):
    run = main.SFT_RUNS_DIR / ("repair-20261116-" + name)
    final = run / "models" / main.SFT_FINAL_MODEL_NAME
    if final.exists():
        raise RuntimeError(f"Run deja present, choisir un nouveau nom: {run}")
    initial = (main.SFT_RUNS_DIR/("repair-20261116-"+init_run)/"models"/main.SFT_FINAL_MODEL_NAME
               if init_run else main._base_model_path())
    model, _ = load_model_from_final_state(initial, tok, str(device))
    train_path,val_path = mixed_data(mix_old,tok) if mix_old else (OUT/"sft_train.npz",OUT/"sft_val.npz")
    train_data, val_data = SFTDataset(train_path), SFTDataset(val_path)
    train_loader = DataLoader(train_data, batch_size=32,
        sampler=ResumableRandomSampler(len(train_data), seed=main.SEED), pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, pin_memory=True)
    config = SFTConfig(epochs=epochs, learning_rate=lr, device=str(device), seed=main.SEED,
        grad_accum_steps=2, warmup_ratio=.02, weight_decay=.01, grad_clip=1.,
        max_cpu_threads=12, max_cpu_interop_threads=6, log_interval=100,
        checkpoint_every_steps=0, resume_from_checkpoint=False, eval_max_batches=None,
        checkpoint_dir=str(run/"checkpoints"), checkpoint_name=main.SFT_CHECKPOINT_NAME,
        final_model_dir=str(run/"models"), final_model_name=main.SFT_FINAL_MODEL_NAME,
        benchmark_path=str(run/"training_benchmark.csv"),
        tokenizer_fingerprint=tok.fingerprint(),
        unlikelihood_exclude_token_ids=tuple(tok.get_token_to_id()[t] for t in tok.special_tokens),
        data_identity={"train_sha256":sha256_file(train_path), "val_sha256":sha256_file(val_path),
                       "initial_model":str(initial),"initial_model_sha256":sha256_file(initial)})
    atomic_json(run/"config.json", asdict(config))
    history = train_sft_model(model, train_loader, config, val_dataloader=val_loader)
    atomic_json(run/"history.json", history)
    del model
    torch.cuda.empty_cache()
    return final


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "baseline", "train", "test"))
    parser.add_argument("--name", default="curriculum1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mix-old", type=int, default=0)
    parser.add_argument("--init-run", help="Optional earlier repair SFT; a fresh optimizer is always used")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_-]+", args.name):
        parser.error("Nom de run invalide")
    if args.init_run and not re.fullmatch(r"[a-z0-9_-]+", args.init_run):
        parser.error("Nom du run initial invalide")
    protected = protect()
    if args.stage == "build":
        build()
    else:
        device = main._setup_device()
        if device.type != "cuda":
            raise RuntimeError("CUDA required")
        torch.set_num_threads(12)
        torch.set_num_interop_threads(6)
        tok = Tokenizer.load(main.TOKENIZER_MODEL_PATH)
        rows = read_lines(OUT / ("test_locked.jsonl" if args.stage == "test" else "dev_generation.jsonl"))
        if args.stage == "baseline":
            score(protected["original_sft"], rows, tok, device, OUT/"baseline_dev.json")
        else:
            path = (train(args.name,args.epochs,args.lr,device,tok,args.mix_old,args.init_run) if args.stage == "train"
                    else main.SFT_RUNS_DIR/("repair-20261116-"+args.name)/"models"/main.SFT_FINAL_MODEL_NAME)
            score(path, rows, tok, device, OUT/(args.name+("_test.json" if args.stage == "test" else "_dev.json")))
    protect()
    print("[repair] Base weights, checkpoint, tokenizer and original SFT data unchanged.", flush=True)


if __name__ == "__main__":
    cli()
