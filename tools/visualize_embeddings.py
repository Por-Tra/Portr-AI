"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @ChatGPT (LLM)

File name : visualize_embeddings.py

Visualisation des embeddings de artifacts/models/gpt_final_state.pt avec
tokenizer/token/bpe_model.json.

Exemples :
    python visualize_embeddings.py
    python visualize_embeddings.py --method umap --dims 2
    python visualize_embeddings.py --method pca --dims 3 --max-points 5000

Dépendances :
    pip install torch numpy scikit-learn plotly
Pour UMAP :
    pip install umap-learn
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise les embeddings des tokens.")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "artifacts/models/gpt_final_state.pt"),
                        help="Chemin du checkpoint PyTorch.")
    parser.add_argument(
        "--tokenizer",
        default=str(PROJECT_ROOT / "tokenizer/token/bpe_model.json"),
                        help="Chemin du tokenizer BPE JSON.")
    parser.add_argument("--method", choices=["pca", "umap"], default="pca",
                        help="Méthode de réduction de dimension (défaut : PCA).")
    parser.add_argument("--dims", type=int, choices=[2, 3], default=2,
                        help="Projection en 2D ou en 3D.")
    parser.add_argument("--max-points", type=int, default=16000,
                        help="Nombre maximal de tokens à afficher (défaut : tous).")
    parser.add_argument("--output", default=None,
                        help="Fichier HTML de sortie.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer introuvable : {tokenizer_path}")

    print(f"Chargement du checkpoint : {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    key = "embedding.token_emb.weight"
    if key not in state:
        raise KeyError(
            f"Clé '{key}' absente. Clés disponibles : {list(state.keys())[:15]}"
        )

    embeddings = state[key].detach().float().cpu().numpy()

    print(f"Chargement du tokenizer : {tokenizer_path}")
    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    token_to_id = tokenizer.get("token_to_id")
    if isinstance(token_to_id, dict):
        tokens = [""] * embeddings.shape[0]
        for token, token_id in token_to_id.items():
            if isinstance(token_id, int) and 0 <= token_id < len(tokens):
                tokens[token_id] = token
        for i, token in enumerate(tokens):
            if not token:
                tokens[i] = f"<token_id_{i}>"
    else:
        vocab = tokenizer.get("vocab", [])
        if len(vocab) != embeddings.shape[0]:
            raise ValueError(
                "Le tokenizer ne contient pas de token_to_id exploitable "
                "et la taille de vocab ne correspond pas aux embeddings."
            )
        tokens = vocab

    if len(tokens) != embeddings.shape[0]:
        raise ValueError(
            f"Incompatibilité : {len(tokens)} tokens pour "
            f"{embeddings.shape[0]} vecteurs."
        )

    n = min(max(1, args.max_points), len(tokens))

    X = embeddings[:n]
    labels = tokens[:n]

    print(f"Embeddings : {embeddings.shape}; tokens visualisés : {n}")
    print(f"Réduction en {args.dims}D avec {args.method.upper()}...")

    if args.method == "pca":
        reducer = PCA(n_components=args.dims, random_state=42)
        coords = reducer.fit_transform(X)
        explained = float(reducer.explained_variance_ratio_.sum() * 100)
        print(f"Variance expliquée par les {args.dims} axes : {explained:.2f}%")
        title = f"Embeddings des tokens — PCA {args.dims}D ({n} tokens)"
    else:
        try:
            import umap
        except ImportError as exc:
            raise SystemExit(
                "UMAP n'est pas installé. Lance : pip install umap-learn"
            ) from exc
        reducer = umap.UMAP(
            n_components=args.dims,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        coords = reducer.fit_transform(X)
        title = f"Embeddings des tokens — UMAP {args.dims}D ({n} tokens)"

    output = args.output or f"embeddings_{args.method}_{args.dims}d.html"

    if args.dims == 2:
        fig = go.Figure(
            data=go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                text=labels,
                customdata=np.arange(n),
                marker=dict(size=5, opacity=0.7),
                hovertemplate=(
                    "<b>Token :</b> %{text}<br>"
                    "<b>ID :</b> %{customdata}<br>"
                    "X : %{x:.3f}<br>Y : %{y:.3f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(xaxis_title="Dimension 1", yaxis_title="Dimension 2")
    else:
        fig = go.Figure(
            data=go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                text=labels,
                customdata=np.arange(n),
                marker=dict(size=3, opacity=0.75),
                hovertemplate=(
                    "<b>Token :</b> %{text}<br>"
                    "<b>ID :</b> %{customdata}<br>"
                    "X : %{x:.3f}<br>Y : %{y:.3f}<br>Z : %{z:.3f}"
                    "<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="closest",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.write_html(output, include_plotlyjs=True)
    print(f"Terminé ! Ouvre ce fichier dans ton navigateur : {Path(output).resolve()}")


if __name__ == "__main__":
    main()
