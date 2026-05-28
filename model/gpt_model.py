"""
gpt_model.py

Implémentation complète du LLM style GPT.

Architecture :

    Tokens [batch, seq]
    |
    Embedding (Token + Position + LayerNorm + Dropout)
    [batch, seq, embedding_dim]
    |
    N x TransformerBlock (Attention + MLP + Résiduel)
    [batch, seq, embedding_dim]
    |
    Final LayerNorm
    [batch, seq, embedding_dim]
    |
    Output Projection  ← poids partagés avec Token Embedding (weight tying)
    [batch, seq, vocab_size]

Optimisations vs version précédente :
------------------------------------------------------------------------------
| Problème                        | Solution                                 |
|---------------------------------|------------------------------------------|
| generate() refait un forward    | KV Cache : seul le dernier token est     |
| O(T) complet à chaque token     | projeté à chaque step -> O(1) par step    |
| output_head et token_emb ont    | Weight tying : une seule matrice         |
| deux matrices distinctes        | partagée -> moins de paramètres,          |
| (vocab_size x dim chacune)      | meilleure généralisation                 |
| Initialisation par défaut       | Scaled init (std=0.02) + résiduel        |
| PyTorch (std=1/√fan_in)         | scaled (std=0.02/√2N) — style GPT-2     |
------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

try:
    from .embedding import Embedding
    from .transformer_block import TransformerBlock
except ImportError:
    from embedding import Embedding
    from transformer_block import TransformerBlock


class GPT(nn.Module):
    """
    Modèle de langage de type GPT.

    Nouveautés :
    - Weight tying : output_head.weight = token_emb.weight
    - KV Cache dans generate() : inférence token-par-token en O(1)
    - Scaled init des couches résiduelles (style GPT-2)
    """

    def __init__(self, vocab_size: int, seq_length: int, embedding_dim: int = 256, num_heads: int = 8, num_blocks: int = 4, dropout: float = 0.1):
        """
        Args:
            vocab_size   : Nombre de tokens uniques dans le vocabulaire
            seq_length   : Longueur maximale des séquences
            embedding_dim: Dimension des embeddings et hidden states
            num_heads    : Nombre de têtes d'attention dans chaque bloc
            num_blocks   : Nombre de blocs Transformer à empiler
            dropout      : Taux de dropout
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        # -- Embedding ----------------------------------------------------
        self.embedding = Embedding(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # -- Stack de blocs Transformer -----------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_length=seq_length,
            )
            for _ in range(num_blocks)
        ])

        # -- Final LayerNorm + projection ---------------------------------
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # -- Weight tying -------------------------------------------------
        # output_head et token_emb partagent la même matrice de poids.
        # - Réduit les paramètres de vocab_size x embedding_dim
        # - Améliore la cohérence sémantique entre entrée et sortie
        self.output_head.weight = self.embedding.token_emb.weight

        # -- Scaled init (style GPT-2) -------------------------------------
        # Les couches de projection résiduelles (out_proj, fc2) sont
        # initialisées avec std = 0.02 / √(2 x num_blocks) pour éviter
        # l'explosion du signal à l'init sur des réseaux profonds.
        self._apply_scaled_init()

    def _apply_scaled_init(self) -> None:
        """Applique l'init GPT-2 avec scaling sur projections residuelles."""
        # 1. Marquer les projections résiduelles
        for block in self.blocks:
            block.attention.out_proj._is_residual_proj = True
            block.mlp.fc2._is_residual_proj = True

        # 2. Appliquer l'init
        std = 0.02 / math.sqrt(2 * self.num_blocks)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_is_residual_proj', False):
                    nn.init.normal_(module.weight, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, seq_length] — indices des tokens

        Returns:
            logits: [batch_size, seq_length, vocab_size]
        """
        x = self.embedding(tokens)

        for block in self.blocks:
            x, _ = block(x, use_cache=False)

        x = self.final_ln(x)
        return self.output_head(x)

    # ------------------------------------------------------------------
    # Génération avec KV Cache
    # ------------------------------------------------------------------

    def generate(
        self,
        initial_tokens: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Génère des tokens avec KV Cache.

        À chaque step, seul le dernier token est projeté (T=1).
        Les K et V des tokens précédents sont lus depuis le cache.
        -> Complexité O(1) par token au lieu de O(T).

        Args:
            initial_tokens: [batch_size, prompt_len]
            max_length    : Nombre total de tokens à générer
            temperature   : Contrôle la diversité (< 1.0 = plus conservateur)
            top_k         : Si spécifié, sample parmi les top-k tokens

        Returns:
            [batch_size, max_length]
        """
        self.eval()
        current_tokens = initial_tokens.clone()

        # -- Prefill : encode le prompt complet, construit le KV cache ----
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        with torch.no_grad():
            # --- Prefill ---
            input_tokens = current_tokens[:, -self.seq_length:]
            if max_length > self.seq_length:
                raise ValueError(
                    f"max_length={max_length} depasse seq_length={self.seq_length}."
                )
            x = self.embedding(input_tokens)

            past_kvs = []
            for block in self.blocks:
                x, kv = block(x, past_kv=None, use_cache=True)
                past_kvs.append(kv)  # type: ignore[arg-type]

            x = self.final_ln(x)
            next_logits = self.output_head(x[:, -1, :])  # [B, vocab]

            # Décode le premier token issu du prefill
            next_token = self._sample(next_logits, temperature, top_k)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            current_pos = input_tokens.shape[1]

            # --- Decode loop (T=1 à chaque step) ---
            while current_tokens.shape[1] < max_length:
                x = self.embedding(next_token, pos_offset=current_pos)  # [B, 1, D]
                current_pos += 1

                new_kvs = []
                for block, past_kv in zip(self.blocks, past_kvs):
                    x, kv = block(x, past_kv=past_kv, use_cache=True)
                    new_kvs.append(kv)  # type: ignore[arg-type]
                past_kvs = new_kvs

                x = self.final_ln(x)
                next_logits = self.output_head(x[:, -1, :])
                next_token = self._sample(next_logits, temperature, top_k)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)

        return current_tokens

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
    ) -> torch.Tensor:
        """
        Applique température + top-k puis sample un token.

        Args:
            logits     : [batch_size, vocab_size]
            temperature: facteur de mise à l'échelle
            top_k      : nombre de candidats conservés

        Returns:
            [batch_size, 1]
        """
        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k, dim=-1)
            # Masquer tout ce qui est sous le k-ième score
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return torch.argmax(logits, dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

class GPTConfig:
    """Configuration du modèle GPT."""

    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 10_000)
        self.seq_length = kwargs.get("seq_length", 512)
        self.embedding_dim = kwargs.get("embedding_dim", 256)
        self.num_heads = kwargs.get("num_heads", 8)
        self.num_blocks = kwargs.get("num_blocks", 4)
        self.dropout = kwargs.get("dropout", 0.1)

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "seq_length": self.seq_length,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "dropout": self.dropout,
        }