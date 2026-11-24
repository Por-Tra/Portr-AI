#-*- coding: utf-8 -*-
"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT (LLM)

File name : diagnose_sft.py

Offline measurements and isolated SFT experiments; never trains the base model.

Run from the repository root with the project's CUDA-enabled Python.
All outputs live in --out. Existing model weights and the active run are protected
by hashes. Evaluation uses the same examples, pure CE and eval mode for all models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main
from data.artifacts import load_token_cache, sha256_file
from data.dataset import TextDataset
from data.sft_dataset import SFTDataset
from inference.generate import decode_token_ids, generate_from_prompt, load_model_from_final_state
from tokenizer.tokenizer import Tokenizer
from training.loss import repetition_unlikelihood
from training.sft_train import SFTConfig, train_sft_model

SEED = 20261116
SAMPLE_SIZE = 2048
PROBES = [
    ("addition", "Réponds uniquement par le résultat : combien font 2 + 2 ?", "4"),
    ("capitale", "Quelle est la capitale de la France ?", "Paris"),
    ("copie", "Recopie exactement ce mot, sans rien ajouter : bleu", "bleu"),
    ("traduction", "Traduis en français le mot anglais « dog ».", "chien"),
    ("salutation", "Dis bonjour en français en un seul mot.", "Bonjour"),
    ("choix", "Choisis B et réponds seulement par cette lettre. A ou B ?", "B"),
    ("definition", "Explique simplement ce qu'est un arbre.", None),
    ("recette", "Donne trois étapes simples pour préparer une omelette.", None),
    ("contexte_rouge", "La boîte contient une balle rouge. De quelle couleur est la balle ?", "rouge"),
    ("contexte_verte", "La boîte contient une balle verte. De quelle couleur est la balle ?", "verte"),
    ("identite", "Comment t'appelles-tu ?", None),
    ("raisonnement", "Paul a trois pommes et en donne une à Léa. Combien lui en reste-t-il ?", "deux ou 2"),
]
EXPERIMENTS = {
    "clip3": {"learning_rate": 1e-5, "grad_clip": 3.0, "epochs": 2},
    "lr3e5": {"learning_rate": 3e-5, "grad_clip": 1.0, "epochs": 2},
    "long6": {"learning_rate": 1e-5, "grad_clip": 1.0, "epochs": 6},
    "ul1": {"learning_rate": 1e-5, "grad_clip": 1.0, "epochs": 2, "unlikelihood_weight": 1.0},
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def protected_paths():
    return [main._base_model_path(), Path(main.CHECKPOINT_DIR) / main.CHECKPOINT_NAME,
            main._sft_model_path(), main.SFT_ACTIVE_RUN_PATH, Path(main.TOKENIZER_MODEL_PATH),
            main.SFT_TRAIN_PATH, main.SFT_VAL_PATH, main.SFT_META_PATH]


def manifest(out):
    path = out / "manifest.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for filename, expected in data["protected_sha256"].items():
            if sha256_file(filename) != expected:
                raise RuntimeError(f"Protected input changed: {filename}")
        return data
    data = {
        "seed": SEED, "train_sample_size": SAMPLE_SIZE,
        "base_model": str(main._base_model_path().resolve()),
        "existing_sft": str(main._sft_model_path().resolve()),
        "protected_sha256": {str(p.resolve()): sha256_file(p) for p in protected_paths()},
        "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
        "experiments": EXPERIMENTS,
    }
    write_json(path, data)
    return data


def subset(dataset):
    ids = np.random.default_rng(SEED).choice(len(dataset), min(SAMPLE_SIZE, len(dataset)), replace=False)
    return Subset(dataset, sorted(ids.tolist()))


def loader(dataset):
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)


def pieces(x, y, assistant_id):
    start = int(np.flatnonzero(y != -100)[0])
    if x[start] != assistant_id:
        raise ValueError("First supervised label must follow the assistant marker")
    return x[1:start].tolist(), y[y != -100].tolist()


def audit_dataset(train, val, tokenizer):
    assistant = tokenizer.get_token_to_id()["<|assistant|>"]
    end = tokenizer.get_token_to_id()["<|end|>"]
    result, hashes, prompts = {}, {}, {}
    for name, ds in (("train", train), ("validation", val)):
        pairs = [pieces(x, y, assistant) for x, y in zip(ds.input_ids, ds.labels)]
        hashes[name] = [hashlib.sha256(np.asarray(p + [-1] + a, dtype="int32").tobytes()).hexdigest() for p, a in pairs]
        prompts[name] = [tuple(p) for p, _ in pairs]
        lengths = np.array([len(a) for _, a in pairs])
        result[name] = {
            "examples": len(ds), "unique_pairs": len(set(hashes[name])),
            "duplicate_pairs": len(ds) - len(set(hashes[name])),
            "unique_prompts": len(set(prompts[name])),
            "answer_tokens_quantiles": np.quantile(lengths, [0, .25, .5, .75, 1]).tolist(),
            "supervised_tokens": int(lengths.sum()),
            "all_answers_end_correctly": all(a[-1] == end for _, a in pairs),
        }
    train_pairs, train_prompts = set(hashes["train"]), set(prompts["train"])
    result["validation_pair_overlap"] = sum(h in train_pairs for h in hashes["validation"])
    overlap = [i for i, p in enumerate(prompts["validation"]) if p in train_prompts]
    result["validation_prompt_overlap"] = len(overlap)
    result["validation_prompt_overlap_indices"] = overlap
    result["validation_examples"] = [
        {"index": int(i), "prompt": decode_token_ids(pieces(val.input_ids[i], val.labels[i], assistant)[0], tokenizer),
         "answer": decode_token_ids(pieces(val.input_ids[i], val.labels[i], assistant)[1], tokenizer)}
        for i in np.random.default_rng(SEED).choice(len(val), min(8, len(val)), replace=False)
    ]
    return result


def shuffled_prompts(val, tokenizer):
    """Negative control: genuine donor prompts, same answer tokens and template.

    Donor prompts may be shortened to retain the full original answer. This is
    a conditioning diagnostic, not a general instruction-following score.
    """
    ids = tokenizer.get_token_to_id()
    parsed = [pieces(x, y, ids["<|assistant|>"]) for x, y in zip(val.input_ids, val.labels)]
    order = np.random.default_rng(SEED).permutation(len(val))
    donors = np.empty(len(val), dtype=int)
    donors[order] = np.roll(order, 1)
    result = []
    for i, (_, answer) in enumerate(parsed):
        prompt = parsed[donors[i]][0][:val.input_ids.shape[1] - len(answer) - 2]
        tokens = [ids["<|user|>"], *prompt, ids["<|assistant|>"], *answer]
        x = np.full(val.input_ids.shape[1], ids["<PAD>"], dtype="int64")
        y = np.full_like(x, -100)
        x[:len(tokens)] = tokens
        start = len(prompt) + 1
        y[start:start + len(answer)] = answer
        result.append((torch.from_numpy(x), torch.from_numpy(y)))
    return result


@torch.inference_mode()
def measure(model, dataset, device, end_id, detailed=False, ul_ids=()):
    model.eval()
    sums, counts = [], []
    correct = total = end_correct = ends = first_correct = 0
    ul_sum = 0.0
    started = time.perf_counter()
    for x, y in loader(dataset):
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        # CE reduced on GPU, saved per example for paired uncertainty estimates.
        losses = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none", ignore_index=-100)
        mask = y != -100
        batch_counts = mask.sum(1)
        sums.extend(losses.sum(1).cpu().tolist())
        counts.extend(batch_counts.cpu().tolist())
        if detailed:
            pred = logits.argmax(-1)
            correct += int(((pred == y) & mask).sum())
            total += int(mask.sum())
            end_mask = y == end_id
            end_correct += int(((pred == y) & end_mask).sum())
            ends += int(end_mask.sum())
            first = mask.long().argmax(1)
            first_correct += int((pred.gather(1, first[:, None]) == y.gather(1, first[:, None])).sum())
            ul = repetition_unlikelihood(logits, y, exclude_token_ids=ul_ids)
            ul_sum += float(ul) * int(mask.sum())
    ce = sum(sums) / sum(counts)
    result = {"ce": ce, "perplexity": math.exp(ce), "examples": len(sums),
              "tokens": sum(counts), "seconds": time.perf_counter() - started}
    if detailed:
        result.update(token_accuracy=correct / total, end_accuracy=end_correct / ends,
                      first_token_accuracy=first_correct / len(sums), unlikelihood=ul_sum / total)
    return result, np.asarray(sums), np.asarray(counts)


def bootstrap_delta(sums_a, sums_b, counts):
    # Same examples and denominator; negative means A has lower CE than B.
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(counts), size=(2000, len(counts)))
    d = (sums_a - sums_b)[indices].sum(1) / counts[indices].sum(1)
    return {"delta_ce": float((sums_a.sum() - sums_b.sum()) / counts.sum()),
            "ci95": np.quantile(d, [.025, .975]).tolist(),
            "unit": "paired bootstrap over validation examples, token-weighted CE"}


def repeat_fraction(ids, n=4):
    grams = [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]
    return (len(grams) - len(set(grams))) / max(1, len(grams))


@torch.inference_mode()
def generations(model, tokenizer, val, device):
    result = []
    assistant = tokenizer.get_token_to_id()["<|assistant|>"]
    end = tokenizer.get_token_to_id()["<|end|>"]
    probes = [(name, text, expected, None) for name, text, expected in PROBES]
    for i in np.random.default_rng(SEED).choice(len(val), min(6, len(val)), replace=False):
        p, a = pieces(val.input_ids[i], val.labels[i], assistant)
        probes.append((f"heldout_{i}", None, None, (p, a)))
    for j, (name, text, expected, raw) in enumerate(probes):
        for temperature in (0.0, 0.9):
            torch.manual_seed(SEED + j)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if raw is None:
                    token_ids, answer = generate_from_prompt(model, tokenizer, text, max_new_tokens=128,
                                                            temperature=temperature, top_k=20,
                                                            device=str(device), use_sft_template=True)
                    prompt_len = len(tokenizer.encode(text)) + 2
                    generated = token_ids[prompt_len:]
                    prompt = text
                    reference = None
                else:
                    p, a = raw
                    initial = [tokenizer.get_token_to_id()["<|user|>"], *p, assistant]
                    out = model.generate(torch.tensor([initial], device=device), len(initial) + 128,
                                         temperature, 20, end)
                    generated = out[0, len(initial):].cpu().tolist()
                    answer = decode_token_ids(generated, tokenizer)
                    prompt, reference = decode_token_ids(p, tokenizer), decode_token_ids(a, tokenizer)
            result.append({"name": name, "prompt": prompt, "reference": reference,
                           "expected": expected, "temperature": temperature, "answer": answer,
                           "tokens": len(generated), "stopped": bool(generated and generated[-1] == end),
                           "repeated_4gram_fraction": repeat_fraction(generated)})
    return result


@torch.inference_mode()
def decoding_checks(model, tokenizer, device):
    """Compare cached decoding with independent full-prefix forward passes."""
    result = []
    end = tokenizer.get_token_to_id()["<|end|>"]
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for _, prompt, _ in PROBES[:3]:
            initial = [tokenizer.get_token_to_id()["<|user|>"], *tokenizer.encode(prompt),
                       tokenizer.get_token_to_id()["<|assistant|>"]]
            prefix = torch.tensor([initial], device=device)
            cached = model.generate(prefix, len(initial) + 64, 0.0, 20, end)
            plain = prefix.clone()
            for _ in range(64):
                next_id = model(plain)[:, -1].argmax(-1, keepdim=True)
                plain = torch.cat([plain, next_id], 1)
                if int(next_id[0, 0]) == end:
                    break
            result.append({"prompt": prompt, "identical_tokens": torch.equal(cached, plain),
                           "cached": decode_token_ids(cached[0, len(initial):].tolist(), tokenizer),
                           "without_cache": decode_token_ids(plain[0, len(initial):].tolist(), tokenizer)})
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    continuations = []
    for prompt in ("La capitale de la France est", "Un arbre est une plante qui", "Pour faire une omelette, il faut"):
        for temperature in (0.0, 0.9):
            torch.manual_seed(SEED)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, text = generate_from_prompt(model, tokenizer, prompt, max_new_tokens=96,
                                               temperature=temperature, top_k=20, device=str(device),
                                               use_sft_template=False)
            continuations.append({"prompt": prompt, "temperature": temperature, "answer": text})
    return {"cache_vs_full_forward_fp32": result, "plain_corpus_continuations": continuations}


@torch.inference_mode()
def objective_breakdown(model, dataset, device, ul_ids):
    """Separate smoothing, dropout, Z and UL on the same 512 corpus windows."""
    result = {}
    indices = np.random.default_rng(SEED).choice(len(dataset), min(512, len(dataset)), replace=False).tolist()
    for dropout in (False, True):
        model.train(dropout)
        torch.manual_seed(SEED)
        values = np.zeros(4)
        total = 0
        for x, y in loader(Subset(dataset, indices)):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            scores = logits.float()
            ce = F.cross_entropy(scores.transpose(1, 2), y, ignore_index=-100)
            smoothed = F.cross_entropy(scores.transpose(1, 2), y, ignore_index=-100,
                                       label_smoothing=main.LABEL_SMOOTHING)
            z = main.Z_LOSS_WEIGHT * scores.logsumexp(-1).square().mean()
            ul = main.UNLIKELIHOOD_WEIGHT * repetition_unlikelihood(logits, y, exclude_token_ids=ul_ids)
            weight = int((y != -100).sum())
            values += np.array([float(ce), float(smoothed), float(z), float(ul)]) * weight
            total += weight
        ce, smooth, z, ul = (values / total).tolist()
        result["dropout_on" if dropout else "dropout_off"] = {
            "ce": ce, "ce_with_smoothing": smooth, "smoothing_increment": smooth - ce,
            "weighted_z": z, "weighted_unlikelihood": ul, "total_training_objective": smooth + z + ul,
            "tokens": total, "examples": min(512, len(dataset)), "gradients_computed": False}
    model.eval()
    return result


def supplementary_checks(out, data, tokenizer, train, val, device):
    write_json(out / "data_audit.json", audit_dataset(train, val, tokenizer))
    corpus = TextDataset(load_token_cache(main.PREPARED_CACHE_DIR / "train_tokens.json", tokenizer), main.SEQ_LENGTH)
    results = {}
    ul_ids = tuple(tokenizer.get_token_to_id()[t] for t in tokenizer.special_tokens)
    for name, path in (("base", data["base_model"]), ("existing_sft", data["existing_sft"])):
        print(f"[diagnostic] decoding and objective checks: {name}", flush=True)
        model, _ = load_model_from_final_state(path, tokenizer, str(device))
        results[name] = decoding_checks(model, tokenizer, device)
        results[name]["corpus_objective_breakdown"] = objective_breakdown(model, corpus, device, ul_ids)
        del model
        torch.cuda.empty_cache()
    write_json(out / "supplementary_checks.json", results)


def evaluate_model(name, path, out, tokenizer, train, val, device, corpus=True):
    dest = out / "measurements" / name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "metrics.json").exists():
        print(f"[diagnostic] already measured: {name}", flush=True)
        return
    print(f"[diagnostic] measuring {name}: {path}", flush=True)
    model, payload = load_model_from_final_state(path, tokenizer, str(device))
    end = tokenizer.get_token_to_id()["<|end|>"]
    ul_ids = tuple(tokenizer.get_token_to_id()[t] for t in tokenizer.special_tokens)
    result = {"model": str(Path(path).resolve()), "sha256": sha256_file(path),
              "saved_config": payload.get("config"), "hparams": payload.get("model_hparams")}
    arrays = {}
    sets = [("sft_validation", val, True), ("sft_train_sample", subset(train), True),
            ("sft_shuffled_prompts", shuffled_prompts(val, tokenizer), False)]
    if corpus:
        corpus_val = TextDataset(load_token_cache(main.PREPARED_CACHE_DIR / "val_tokens.json", tokenizer), main.SEQ_LENGTH)
        corpus_train = TextDataset(load_token_cache(main.PREPARED_CACHE_DIR / "train_tokens.json", tokenizer), main.SEQ_LENGTH)
        sets.extend([("corpus_validation_full", corpus_val, False), ("corpus_train_sample", subset(corpus_train), False)])
    for label, ds, detailed in sets:
        metrics, sums, counts = measure(model, ds, device, end, detailed, ul_ids)
        result[label] = metrics
        arrays[label + "_sums"], arrays[label + "_counts"] = sums, counts
        print(f"[diagnostic] {name} {label}: CE={metrics['ce']:.6f}; tokens={metrics['tokens']}", flush=True)
        if label == "corpus_validation_full":
            cut = 200 * 32
            for part, s, c in (("first_200_batches", sums[:cut], counts[:cut]), ("remaining_batches", sums[cut:], counts[cut:])):
                result[label][part] = {"ce": float(s.sum() / c.sum()), "tokens": int(c.sum())}
    result["shuffled_prompt_delta"] = bootstrap_delta(arrays["sft_shuffled_prompts_sums"], arrays["sft_validation_sums"],
                                                     arrays["sft_validation_counts"])
    samples = generations(model, tokenizer, val, device)
    write_json(dest / "generations.json", samples)
    result["generation_summary"] = {
        str(t): {"stopped": sum(x["stopped"] for x in samples if x["temperature"] == t),
                 "n": sum(x["temperature"] == t for x in samples),
                 "mean_repeated_4gram_fraction": float(np.mean([x["repeated_4gram_fraction"] for x in samples if x["temperature"] == t]))}
        for t in (0.0, 0.9)
    }
    np.savez_compressed(dest / "per_example.npz", **arrays)
    write_json(dest / "metrics.json", result)
    del model, payload
    torch.cuda.empty_cache()


def run_experiment(name, out, data, tokenizer, device):
    if name not in EXPERIMENTS:
        raise ValueError(name)
    dest = out / "experiments" / name
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "models" / main.SFT_FINAL_MODEL_NAME
    if final.exists():
        return final
    # Uses SFT training only, always initialized from the protected base weights.
    torch.manual_seed(main.SEED)
    model, _ = load_model_from_final_state(data["base_model"], tokenizer, str(device))
    train_loader, val_loader = main._build_sft_dataloaders(device)
    overrides = dict(EXPERIMENTS[name])
    ul_weight = overrides.pop("unlikelihood_weight", main.SFT_UNLIKELIHOOD_WEIGHT)
    config = SFTConfig(
        **overrides, device=str(device), weight_decay=main.SFT_WEIGHT_DECAY,
        seed=main.SEED, log_interval=main.LOG_INTERVAL, max_cpu_threads=main.MAX_CPU_THREADS,
        max_cpu_interop_threads=main.MAX_CPU_INTEROP_THREADS, grad_accum_steps=main.GRAD_ACCUM_STEPS_CUDA,
        checkpoint_dir=str(dest / "checkpoints"), checkpoint_name=main.SFT_CHECKPOINT_NAME,
        checkpoint_every_steps=0, resume_from_checkpoint=False,
        final_model_dir=str(dest / "models"), final_model_name=main.SFT_FINAL_MODEL_NAME,
        benchmark_path=str(dest / "training_benchmark.csv"),
        warmup_ratio=main.WARMUP_RATIO, lr_decay_factor=main.LR_DECAY_FACTOR,
        use_amp=True, amp_dtype=main.AMP_DTYPE,
        eval_interval=100, eval_max_batches=None,
        label_smoothing=main.SFT_LABEL_SMOOTHING, z_loss_weight=main.SFT_Z_LOSS_WEIGHT,
        unlikelihood_weight=ul_weight,
        unlikelihood_ngram_size=main.UNLIKELIHOOD_NGRAM_SIZE,
        unlikelihood_window=main.UNLIKELIHOOD_WINDOW,
        unlikelihood_exclude_token_ids=tuple(tokenizer.get_token_to_id()[t] for t in tokenizer.special_tokens),
        tokenizer_fingerprint=tokenizer.fingerprint(),
        data_identity={"base_model": data["protected_sha256"][data["base_model"]]},
    )
    write_json(dest / "config.json", vars(config))
    history = train_sft_model(model, train_loader, config, val_dataloader=val_loader)
    write_json(dest / "history.json", history)
    del model
    torch.cuda.empty_cache()
    return final


def summary(out):
    rows, gens, comparisons = [], {}, {}
    audit = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))
    overlap = audit.get("validation_prompt_overlap_indices")
    reference = out / "measurements" / "existing_sft" / "per_example.npz"
    base = out / "measurements" / "base" / "per_example.npz"
    for p in sorted((out / "measurements").glob("*/metrics.json")):
        name = p.parent.name
        m = json.loads(p.read_text(encoding="utf-8"))
        row = {"model": name}
        for key in ("sft_validation", "sft_train_sample", "corpus_validation_full", "corpus_train_sample"):
            row[key + "_ce"] = m.get(key, {}).get("ce")
        row["sft_end_accuracy"] = m["sft_validation"]["end_accuracy"]
        row["shuffled_prompt_delta"] = m["shuffled_prompt_delta"]["delta_ce"]
        rows.append(row)
        gens[name] = json.loads((p.parent / "generations.json").read_text(encoding="utf-8"))
        a = np.load(p.parent / "per_example.npz")
        comparisons[name] = {}
        if overlap is not None:
            for group, indices in (("seen_prompts", overlap),
                                   ("unseen_prompts", sorted(set(range(len(a["sft_validation_counts"]))) - set(overlap)))):
                if indices:
                    s, c = a["sft_validation_sums"][indices], a["sft_validation_counts"][indices]
                    comparisons[name][group] = {"examples": len(indices), "ce": float(s.sum() / c.sum())}
        for label, ref in (("vs_existing_sft", reference), ("vs_base", base)):
            if ref.exists():
                b = np.load(ref)
                comparisons[name][label] = bootstrap_delta(a["sft_validation_sums"], b["sft_validation_sums"], a["sft_validation_counts"])
                if overlap is not None:
                    unseen = sorted(set(range(len(a["sft_validation_counts"]))) - set(overlap))
                    comparisons[name][label + "_unseen_prompts"] = bootstrap_delta(
                        a["sft_validation_sums"][unseen], b["sft_validation_sums"][unseen], a["sft_validation_counts"][unseen])
    write_json(out / "comparison.json", {"metrics": rows, "paired_comparisons": comparisons})
    if rows:
        with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = ["# Comparaison des générations", "", "Même question et même graine par modèle. Les réponses ne sont pas notées automatiquement.", ""]
    for i in range(len(next(iter(gens.values()), []))):
        first = next(iter(gens.values()))[i]
        text.extend([f"## {first['name']} — température {first['temperature']}", "", first["prompt"], ""])
        if first["reference"]:
            text.extend(["Référence : " + first["reference"], ""])
        for name, samples in gens.items():
            text.extend([f"**{name}** : {samples[i]['answer']}", ""])
    (out / "generations.md").write_text("\n".join(text), encoding="utf-8")
    plot_summary(out, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


def plot_summary(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"base": "Base", "existing_sft": "SFT actuel", "clip3": "Clipping 3",
             "lr3e5": "LR 3e-5", "long6": "6 époques", "ul1": "Unlikelihood 1"}
    palette = {"base": "#666666", "existing_sft": "#2463b4", "clip3": "#8b65b5",
               "lr3e5": "#d37125", "long6": "#21836b", "ul1": "#b24477"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    for row in rows:
        name = row["model"]
        if name == "base":
            axes[0].axhline(row["sft_validation_ce"], color=palette[name], ls="--", label="Base avant SFT")
            continue
        benchmark = (Path(json.loads((out / "manifest.json").read_text())["existing_sft"]).parents[1] / "training_benchmark.csv"
                     if name == "existing_sft" else out / "experiments" / name / "training_benchmark.csv")
        with benchmark.open(encoding="utf-8", newline="") as f:
            val = [r for r in csv.DictReader(f) if r.get("event") == "validation"]
        axes[0].plot([int(r["step"]) for r in val], [float(r["val_loss"]) for r in val],
                     marker="o", ms=3, label=names[name], color=palette[name])
    axes[0].set(title="Prédiction des références SFT", xlabel="Pas d'optimiseur", ylabel="Cross-entropy de validation")
    axes[0].legend(fontsize=8)
    plotted = [r for r in rows if r["model"] != "base"]
    for i, row in enumerate(plotted):
        m = json.loads((out / "measurements" / row["model"] / "metrics.json").read_text())
        for shift, temp, alpha in ((-.17, "0.0", 1.0), (.17, "0.9", .45)):
            axes[1].bar(i + shift, 100*m["generation_summary"][temp]["mean_repeated_4gram_fraction"],
                        width=.32, color=palette[row["model"]], alpha=alpha,
                        label=("Glouton" if temp == "0.0" else "Température 0,9") if i == 0 else None)
    axes[1].set_xticks(range(len(plotted)), [names[r["model"]] for r in plotted], rotation=25, ha="right")
    axes[1].set(title="Répétition en génération libre", ylabel="4-grammes répétés (%)", ylim=(0, 100))
    axes[1].legend(fontsize=8)
    for row in rows:
        name = row["model"]
        axes[2].scatter(row["sft_validation_ce"], row["corpus_validation_full_ce"], color=palette[name], s=50)
        offset = {"existing_sft": (8, 18), "clip3": (8, -14), "ul1": (8, 3)}.get(name, (5, 5))
        axes[2].annotate(names[name], (row["sft_validation_ce"], row["corpus_validation_full_ce"]),
                         xytext=offset, textcoords="offset points", fontsize=8)
    axes[2].margins(.2)
    axes[2].set(title="Adaptation et rétention du corpus", xlabel="CE SFT (plus bas = mieux)", ylabel="CE corpus complet (plus bas = mieux)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Diagnostic SFT — mêmes exemples et poids de base conservés", fontsize=15)
    fig.savefig(out / "diagnostic_overview.png", dpi=160)
    plt.close(fig)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("baseline", "experiment", "summary", "checks"), required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "artifacts" / "diagnostics").resolve()):
        parser.error("Outputs must be inside artifacts/diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    device = main._setup_device()
    if device.type != "cuda":
        raise RuntimeError("This diagnostic expects the project's CUDA environment")
    torch.set_num_threads(main.MAX_CPU_THREADS)
    torch.set_num_interop_threads(main.MAX_CPU_INTEROP_THREADS)
    data = manifest(out)
    tokenizer = Tokenizer.load(main.TOKENIZER_MODEL_PATH)
    if args.stage == "summary":
        summary(out)
    else:
        train, val = SFTDataset(main.SFT_TRAIN_PATH), SFTDataset(main.SFT_VAL_PATH)
        if args.stage == "baseline":
            write_json(out / "data_audit.json", audit_dataset(train, val, tokenizer))
            evaluate_model("base", data["base_model"], out, tokenizer, train, val, device)
            evaluate_model("existing_sft", data["existing_sft"], out, tokenizer, train, val, device)
        elif args.stage == "checks":
            supplementary_checks(out, data, tokenizer, train, val, device)
        else:
            if not args.experiment:
                parser.error("--experiment is required")
            path = run_experiment(args.experiment, out, data, tokenizer, device)
            evaluate_model(args.experiment, path, out, tokenizer, train, val, device)
        summary(out)
    manifest(out)
    print("[diagnostic] Protected models, active run, tokenizer and SFT data hashes unchanged.", flush=True)


if __name__ == "__main__":
    cli()
