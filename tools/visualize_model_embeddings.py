"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : visualize_model_embeddings.py

Explore learned token/position embeddings and contextual token representations.

python tools/visualize_model_embeddings.py
python tools/visualize_model_embeddings.py --model base --dims 3
python tools/visualize_model_embeddings.py --kind positions
python tools/visualize_model_embeddings.py --kind contextual --text "Le chat dort." --text "Le chat mange."

CPU only by default. No training or checkpoint modification. HTML works offline.
The older tools/visualize_embeddings.py remains available unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import sys
import webbrowser

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.gpt_model import GPT
from tokenizer.tokenizer import Tokenizer


def resolve_model(selection: str) -> Path:
    if selection == "base":
        return ROOT / "artifacts/models/gpt_final_state.pt"
    if selection != "sft":
        return Path(selection).expanduser().resolve()
    pointer = ROOT / "artifacts/sft/active_run.json"
    if not pointer.exists():
        return ROOT / "artifacts/sft/models/sft_final_state.pt"
    run = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"])
    run = (ROOT / run).resolve() if not run.is_absolute() else run.resolve()
    runs = (ROOT / "artifacts/sft/runs").resolve()
    if run == runs or not run.is_relative_to(runs):
        raise ValueError("Le run SFT actif doit appartenir a artifacts/sft/runs")
    return run / "models/sft_final_state.pt"


def load_weights(path: Path, tokenizer: Tokenizer):
    # One immutable byte snapshot: an active training job may replace the file.
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload)
    weights = state.get("embedding.token_emb.weight")
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise ValueError("Table embedding.token_emb.weight absente ou invalide")
    vocab_size = weights.shape[0]
    fingerprint = payload.get("tokenizer_fingerprint")
    if fingerprint and tokenizer.fingerprint(vocab_size=vocab_size) != fingerprint:
        raise ValueError("Tokenizer incompatible avec le modele sauvegarde")
    if set(range(vocab_size)) - tokenizer.get_id_to_token().keys():
        raise ValueError("Certains IDs du modele n'ont pas de token dans ce tokenizer")
    return payload, state, digest


def model_from_snapshot(payload, device):
    hparams = payload.get("model_hparams")
    required = ("vocab_size", "seq_length", "embedding_dim", "num_heads", "num_blocks", "dropout")
    if not isinstance(hparams, dict) or any(key not in hparams for key in required):
        raise ValueError("Le mode contextual requiert un artefact final avec model_hparams complets")
    model = GPT(**{key: hparams[key] for key in required})
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def token_group(token: str, tokenizer: Tokenizer) -> str:
    if token in tokenizer.special_tokens:
        return "Tokens speciaux"
    text = token.replace(tokenizer.end_of_word, "") if tokenizer.end_of_word else token
    text = text.strip("_ ")
    if text.isalpha():
        return "Lettres"
    if text.isdecimal():
        return "Chiffres"
    if text and all(not c.isalnum() for c in text):
        return "Ponctuation"
    return "Mixte / espaces"


@torch.inference_mode()
def contextual_vectors(model, tokenizer: Tokenizer, texts: list[str], device: str):
    """Last hidden states after final_ln, before vocabulary projection. No logits."""
    model.eval()
    vectors, rows = [], []
    for number, text in enumerate(texts, 1):
        ids = tokenizer.encode(text)
        if not ids:
            raise ValueError(f"Texte {number} vide apres tokenisation")
        if len(ids) > model.seq_length:
            raise ValueError(f"Texte {number}: {len(ids)} tokens, contexte limite a {model.seq_length}. Raccourcir le texte.")
        x = model.embedding(torch.tensor([ids], dtype=torch.long, device=device))
        for block in model.blocks:
            x, _ = block(x, use_cache=False)
        vectors.append(model.final_ln(x)[0].float().cpu().numpy())
        for position, token_id in enumerate(ids):
            rows.append({"id": len(rows), "token": tokenizer.id_to_token[token_id],
                         "token_id": token_id, "position": position, "text_number": number,
                         "group": f"Texte {number}"})
    return np.concatenate(vectors), rows


def validate_vectors(vectors):
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Les vecteurs doivent former une matrice finie non vide")
    return np.ascontiguousarray(x)


def normalized_vectors(vectors):
    x = validate_vectors(vectors)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0), norms[:, 0]


def sample_indices(count: int, maximum: int, seed: int):
    if maximum < 0:
        raise ValueError("--max-points doit etre positif ou 0 pour tout afficher")
    if maximum == 0 or maximum >= count:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, replace=False))


def project(vectors, method: str, dims: int, seed: int, normalize: bool):
    x = normalized_vectors(vectors)[0] if normalize else validate_vectors(vectors)
    if min(x.shape) < dims or len(x) <= dims:
        raise ValueError(f"La projection {dims}D requiert au moins {dims + 1} points et {dims} dimensions")
    if float(np.var(x, axis=0).sum()) == 0:
        raise ValueError("Tous les vecteurs sont identiques; aucune projection informative")
    if method == "pca":
        reducer = PCA(n_components=dims, svd_solver="randomized", random_state=seed)
        coordinates = reducer.fit_transform(x)
        info = {"explained_variance_ratio": reducer.explained_variance_ratio_.tolist()}
    else:
        try:
            import umap
        except ImportError as exc:
            raise ValueError("UMAP optionnel absent. Installer umap-learn ou utiliser --method pca.") from exc
        reducer = umap.UMAP(n_components=dims, n_neighbors=min(15, len(x) - 1),
            min_dist=.1, metric="cosine", random_state=seed, n_jobs=1, init="random")
        coordinates = reducer.fit_transform(x)
        info = {}
    return coordinates.astype(np.float32), info


@torch.inference_mode()
def cosine_neighbors(vectors, indices, k: int):
    """Exact cosine in original space, across ALL rows, in bounded memory chunks."""
    unit, norms = normalized_vectors(vectors)
    matrix = torch.from_numpy(unit)
    k = min(k, len(matrix) - 1)
    if k < 1:
        raise ValueError("Au moins deux vecteurs et un voisin sont requis")
    neighbors, scores = [], []
    for start in range(0, len(indices), 128):
        query = np.asarray(indices[start:start + 128])
        similarities = matrix[query] @ matrix.T
        similarities[:, torch.from_numpy(norms == 0)] = -torch.inf
        similarities[torch.arange(len(query)), torch.from_numpy(query)] = -torch.inf
        values, ids = similarities.topk(k, dim=1)
        for row, value, index in zip(query, values.tolist(), ids.tolist()):
            valid = [(i, float(np.clip(s, -1, 1))) for i, s in zip(index, value)
                     if np.isfinite(s) and norms[row] > 0]
            neighbors.append([i for i, _ in valid])
            scores.append([s for _, s in valid])
    return neighbors, scores


def safe_json(value) -> str:
    # Prevent tokens/user texts from closing an HTML script element.
    return (json.dumps(value, ensure_ascii=False, allow_nan=False)
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def write_viewer(path, rows, indices, coordinates, neighbors, scores, metadata):
    dims = coordinates.shape[1]
    trace_class = go.Scattergl if dims == 2 else go.Scatter3d
    figure = go.Figure()
    palette = ["#3178b5", "#17a398", "#8b6bb1", "#d19132", "#687a8d", "#c25a7c"]
    groups = list(dict.fromkeys(rows[int(i)]["group"] for i in indices))
    for color, group in enumerate(groups):
        at = [j for j, i in enumerate(indices) if rows[int(i)]["group"] == group]
        options = {"x": coordinates[at, 0], "y": coordinates[at, 1]}
        if dims == 3:
            options["z"] = coordinates[at, 2]
        figure.add_trace(trace_class(**options, mode="markers", name=group,
            customdata=at, text=[html.escape(rows[int(indices[j])]["token"]) for j in at],
            marker={"size": 4 if dims == 2 else 2.5, "opacity": .65, "color": palette[color % len(palette)]},
            hovertemplate="%{text}<extra>%{fullData.name}</extra>"))
    base_traces = len(figure.data)
    for name, size, color in (("Voisins", 8, "#eeaa53"), ("Selection", 13, "#df6633")):
        options = {"x": [], "y": []}
        if dims == 3:
            options["z"] = []
        figure.add_trace(trace_class(**options, mode="markers", name=name, showlegend=False,
            marker={"size": size if dims == 2 else size / 2, "color": color, "line": {"color": "white", "width": 1}},
            hoverinfo="skip"))
    figure.update_layout(template="plotly_white", margin={"l": 45, "r": 20, "t": 20, "b": 45},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font={"family": "Segoe UI, sans-serif", "color": "#526476"},
        legend={"orientation": "h", "y": -.13}, uirevision="embeddings")
    if dims == 2:
        figure.update_xaxes(title="Axe 1", gridcolor="#edf1f5", zeroline=False)
        figure.update_yaxes(title="Axe 2", gridcolor="#edf1f5", zeroline=False)
    else:
        figure.update_layout(scene={f"{a}axis": {"title": f"Axe {i + 1}"} for i, a in enumerate("xyz")})
    chart = figure.to_html(full_html=False, include_plotlyjs=True, div_id="embedding-map",
        config={"responsive": True, "displaylogo": False, "scrollZoom": True}, default_height="100%")
    data = {"rows": rows, "indices": indices.tolist(), "coordinates": coordinates.tolist(),
            "neighbors": neighbors, "scores": scores, "dims": dims, "baseTraces": base_traces}
    title = {"tokens": "Embeddings des tokens", "positions": "Embeddings des positions",
             "contextual": "Representations dans le contexte"}[metadata["kind"]]
    template = (ROOT / "tools/embedding_viewer.html").read_text(encoding="utf-8")
    variance = metadata.get("explained_variance_ratio")
    projection = f"{metadata['method'].upper()} {dims}D"
    if variance:
        projection += f" · {100 * sum(variance):.1f} % de variance"
    replacements = {"{{TITLE}}": html.escape(title), "{{MODEL}}": html.escape(metadata["model_label"]),
        "{{COUNT}}": f"{len(indices):,}".replace(",", " "), "{{TOTAL}}": str(len(rows)),
        "{{FEATURES}}": str(metadata["features"]), "{{CONTEXT}}": str(metadata["context"]),
        "{{PROJECTION}}": html.escape(projection), "{{CHART}}": chart, "{{DATA}}": safe_json(data),
        "{{CHECKPOINT}}": html.escape(metadata["checkpoint"]),
        "{{SNAPSHOT}}": html.escape(metadata.get("checkpoint_sha256", "non disponible")),
        "{{NORMALIZED}}": "Vecteurs normalises L2" if metadata["normalize"] else "Vecteurs bruts"}
    template = re.sub(r"\{\{[A-Z]+\}\}", lambda match: replacements.get(match[0], match[0]), template)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template, encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="sft", help="sft (actif), base, ou chemin d'un artefact .pt")
    parser.add_argument("--checkpoint", dest="model", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer/token/bpe_model.json")
    parser.add_argument("--kind", choices=("tokens", "positions", "contextual"), default="tokens")
    parser.add_argument("--text", action="append", default=[], help="Texte a encoder, option repetable (mode contextual)")
    parser.add_argument("--text-file", type=Path, help="Fichier UTF-8, un texte par ligne non vide")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Encodage contextual seulement")
    parser.add_argument("--method", choices=("pca", "umap"), default="pca")
    parser.add_argument("--dims", type=int, choices=(2, 3), default=2)
    parser.add_argument("--max-points", type=int, default=0, help="0 = tous; sinon echantillon uniforme reproductible")
    parser.add_argument("--neighbors", type=int, default=10, help="Nombre de voisins cosinus, entre 1 et 30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, help="Fichier HTML (defaut: artifacts/visualizations/embeddings/)")
    parser.add_argument("--export-vectors", action="store_true", help="Sauvegarde aussi les vecteurs affiches en NPZ")
    parser.add_argument("--open", action="store_true", help="Ouvre le HTML dans le navigateur")
    args = parser.parse_args(argv)
    if not 1 <= args.neighbors <= 30:
        parser.error("--neighbors doit etre compris entre 1 et 30")
    if args.max_points < 0:
        parser.error("--max-points doit etre positif ou nul")
    if args.text_file:
        args.text.extend(line.strip() for line in args.text_file.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    if args.kind == "contextual" and not args.text:
        parser.error("--kind contextual requiert --text ou --text-file")
    if args.kind != "contextual" and args.text:
        parser.error("--text et --text-file sont reserves au mode contextual")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    checkpoint = resolve_model(args.model).resolve()
    tokenizer = Tokenizer.load(args.tokenizer)
    payload, state, digest = load_weights(checkpoint, tokenizer)
    context = state["embedding.position_emb.weight"].shape[0]
    if args.kind == "contextual":
        model = model_from_snapshot(payload, args.device)
        vectors, rows = contextual_vectors(model, tokenizer, args.text, args.device)
        del model
    elif args.kind == "positions":
        vectors = state["embedding.position_emb.weight"].detach().float().numpy()
        rows = [{"id": i, "token": f"Position {i}", "position": i,
                 "group": "Positions 0-255" if i < 256 else "Positions 256+"} for i in range(len(vectors))]
    else:
        vectors = state["embedding.token_emb.weight"].detach().float().numpy()
        rows = [{"id": i, "token": tokenizer.id_to_token[i], "token_id": i,
                 "group": token_group(tokenizer.id_to_token[i], tokenizer)} for i in range(len(vectors))]
    vectors = validate_vectors(vectors)
    _, norms = normalized_vectors(vectors)
    for row, norm in zip(rows, norms):
        row["norm"] = float(norm)
    indices = sample_indices(len(vectors), args.max_points, args.seed)
    print(f"[embeddings] {checkpoint}\n[embeddings] {vectors.shape}; affichage={len(indices)}; mode={args.kind}", flush=True)
    coordinates, info = project(vectors[indices], args.method, args.dims, args.seed, args.normalize)
    print("[embeddings] Calcul des voisins cosinus dans l'espace original...", flush=True)
    neighbors, scores = cosine_neighbors(vectors, indices, args.neighbors)
    label = args.model.upper() if args.model in ("base", "sft") else checkpoint.stem
    selection = payload.get("selection") or {}
    metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "snapshot_policy": "Single byte snapshot read before extracting vectors; source file may later change",
        "model_selection": {key: selection[key] for key in ("criterion", "epoch", "step", "val_loss") if key in selection},
        "tokenizer": str(args.tokenizer.resolve()), "tokenizer_fingerprint": tokenizer.fingerprint(vocab_size=state['embedding.token_emb.weight'].shape[0]),
        "fingerprint_verified": bool(payload.get("tokenizer_fingerprint")),
        "model_label": label, "kind": args.kind, "context": context, "features": vectors.shape[1],
        "total": len(vectors), "displayed": len(indices), "method": args.method, "dims": args.dims,
        "seed": args.seed, "normalize": args.normalize, "texts": args.text,
        "neighbors": "exact cosine over all original vectors; zero vectors excluded", **info}
    output = (args.output or ROOT / f"artifacts/visualizations/embeddings/{label.lower()}_{args.kind}_{args.method}_{args.dims}d.html").resolve()
    if output.suffix.lower() != ".html":
        raise ValueError("--output doit etre un fichier .html")
    outputs = {output, output.with_suffix(".json"), output.with_suffix(".npz")}
    protected = {checkpoint, args.tokenizer.resolve(), (ROOT / "artifacts/sft/active_run.json").resolve()}
    if outputs & protected:
        raise ValueError("La sortie ne doit pas remplacer un artefact d'entree")
    write_viewer(output, rows, indices, coordinates, neighbors, scores, metadata)
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.export_vectors:
        np.savez_compressed(output.with_suffix(".npz"), vectors=vectors[indices], indices=indices, coordinates=coordinates)
    print(f"[embeddings] HTML hors ligne : {output}", flush=True)
    if args.open:
        webbrowser.open(output.as_uri())
    return output


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[embeddings] {error}") from error
