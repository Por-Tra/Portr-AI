"""Lance les poids publiés, sans importer le pipeline d'entraînement."""

import argparse
import json
from pathlib import Path

import torch

from inference.generate import generate_from_prompt, load_model_from_final_state
from tokenizer.tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base", "sft"), default="sft")
    parser.add_argument("--prompt", help="Une question ; sinon, mode interactif (:quit pour sortir).")
    parser.add_argument("--model-path", type=Path, help="Chemin explicite vers les poids .pt.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, help="Défaut : 0 pour SFT, 0.9 pour la base.")
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens doit être positif ou nul")

    weights = args.model_path
    if weights is None:
        weights = ROOT / ("artifacts/sft/models/sft_final_state.pt" if args.model == "sft"
                          else "artifacts/models/gpt_final_state.pt")
        active = ROOT / "artifacts/sft/active_run.json"
        if args.model == "sft" and active.exists():
            weights = Path(json.loads(active.read_text(encoding="utf-8"))["run_dir"]) / "models/sft_final_state.pt"
    if not weights.is_file():
        parser.error(f"Poids absents : {weights}. Extraire l'archive {args.model} de la release 1.0.1.")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponible ; utiliser --device cpu")
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    tokenizer = Tokenizer.load(ROOT / "tokenizer/token/bpe_model.json")
    model, _ = load_model_from_final_state(weights, tokenizer, device=device)
    temperature = args.temperature if args.temperature is not None else (0.0 if args.model == "sft" else 0.9)

    def answer(prompt):
        _, text = generate_from_prompt(
            model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
            temperature=temperature, top_k=20, device=device,
            use_sft_template=args.model == "sft",
        )
        print(text)

    if args.prompt is not None:
        answer(args.prompt)
    else:
        print(f"Portr-AI {args.model} ({device}) — :quit pour sortir. Chaque prompt est indépendant.")
        while True:
            try:
                prompt = input("prompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt == ":quit":
                break
            if prompt:
                answer(prompt)


if __name__ == "__main__":
    main()
