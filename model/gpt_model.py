# -*- coding: utf-8 -*-
"""
██████╗  ██████╗ ██████╗ ████████╗██████╗        █████╗ ██╗
██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝██╔══██╗      ██╔══██╗██║
██████╔╝██║   ██║██████╔╝   ██║   ██████╔╝█████╗███████║██║
██╔═══╝ ██║   ██║██╔══██╗   ██║   ██╔══██╗╚════╝██╔══██║██║
██║     ╚██████╔╝██║  ██║   ██║   ██║  ██║      ██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝      ╚═╝  ╚═╝╚═╝

auteur : @Lucas Contreras Hodapp

File name : gpt_model.py

description du script:
    
    Implémentation complète du LLM style GPT.

    Architecture :

        1. Tokens [batch, seq]
        
        2. Embedding (Token + Position + Dropout) [batch, seq, embedding_dim]
        
        3. N x TransformerBlock (Attention + MLP + Résiduel) [batch, seq, embedding_dim]
        
        4. Final LayerNorm [batch, seq, embedding_dim]
        
        5. Output Projection  <- poids partagés avec Token Embedding (weight tying) [batch, seq, vocab_size]
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
    - KV Cache dans generate() : seul le nouveau token est projeté à chaque step
    - Scaled init des couches résiduelles (style GPT-2)
    """

    def __init__(self, vocab_size: int, seq_length: int, embedding_dim: int = 256, num_heads: int = 8, num_blocks: int = 4, dropout: float = 0.1):
        """
        Args:
            vocab_size (int): Taille du vocabulaire (nombre de tokens uniques)
            seq_length (int): Longueur maximale des séquences
            embedding_dim (int, optional): Dimension des embeddings et hidden states. Defaults to 256.
            num_heads (int, optional): Nombre de têtes d'attention dans chaque bloc. Defaults to 8.
            num_blocks (int, optional): Nombre de blocs Transformer à empiler. Defaults to 4.
            dropout (float, optional): Taux de dropout. Defaults to 0.1.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        #* Embedding 
        self.embedding = Embedding(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            dropout=dropout
        )

        #* Stack de blocs Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_length=seq_length
            )
            for _ in range(num_blocks)
        ])

        #* Final LayerNorm + projection
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        #* Weight tying
        #& output_head et token_emb partagent la même matrice de poids.
        #& - Réduit les paramètres de vocab_size x embedding_dim
        #& - Améliore la cohérence sémantique entre entrée et sortie
        self.output_head.weight = self.embedding.token_emb.weight

        #* Scaled init (style GPT-2)
        #& Les couches de projection résiduelles (out_proj, fc2) sont
        #& initialisées avec std = 0.02 / √(2 x num_blocks) pour éviter
        #& l'explosion du signal à l'init sur des réseaux profonds.
        self._apply_scaled_init()

    def _apply_scaled_init(self) -> None:
        """Applique l'init GPT-2 avec scaling sur projections residuelles."""
        #& 1. Marquer les projections résiduelles
        for block in self.blocks:
            block.attention.out_proj._is_residual_proj = True
            block.mlp.fc2._is_residual_proj = True

        #& 2. Appliquer l'init
        #&    - projections résiduelles (out_proj, fc2) : std = 0.02 / √(2N)
        #&    - autres Linear (qkv_proj, fc1)          : std = 0.02
        #&      (avant : laissées à l'init kaiming par défaut de PyTorch)
        residual_std = 0.02 / math.sqrt(2 * self.num_blocks)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = residual_std if getattr(module, '_is_residual_proj', False) else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    #* ------------------------------------------------------------------
    #*                      Forward (training)
    #* ------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            tokens (torch.Tensor): indices des tokens d'entrée de forme [batch_size, seq_length]

        Returns:
            torch.Tensor: logits de sortie de forme [batch_size, seq_length, vocab_size]
        """
        x = self.embedding(tokens)

        for block in self.blocks:
            x, _ = block(x, use_cache=False)

        x = self.final_ln(x)
        return self.output_head(x)

    #* ------------------------------------------------------------------
    #*                      Génération avec KV Cache
    #* ------------------------------------------------------------------

    def _prefill(self, input_tokens: torch.Tensor):
        """Encode une séquence complète et construit le KV cache.

        Returns:
            Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int, torch.Tensor] : (past_kvs, position suivante, logits du dernier token [B, vocab])
        """
        x = self.embedding(input_tokens)
        past_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            x, kv = block(x, past_kv=None, use_cache=True)
            past_kvs.append(kv)  # type: ignore[arg-type]
        x = self.final_ln(x)
        return past_kvs, input_tokens.shape[1], self.output_head(x[:, -1, :])

    def generate(self, initial_tokens: torch.Tensor, max_length: int, temperature: float = 1.0, top_k: Optional[int] = None):
        """
        Génère des tokens avec KV Cache.

        À chaque step, seul le dernier token est projeté (T=1).
        Les K et V des tokens précédents sont lus depuis le cache
        (l'attention reste en O(T) par token, mais sans recalculer le passé).

        Quand le contexte atteint seq_length (limite des position embeddings),
        on repart d'un prefill sur la seconde moitié de la fenêtre : la
        génération peut donc dépasser seq_length et le prompt peut être long.

        Args:
            initial_tokens: [batch_size, prompt_len]
            max_length    : Longueur totale visée (prompt inclus)
            temperature   : Contrôle la diversité (< 1.0 = plus conservateur)
            top_k         : Si spécifié, sample parmi les top-k tokens

        Returns:
            torch.Tensor: tokens générés de forme [batch_size, max(max_length, prompt_len)]
        """
        was_training = self.training
        self.eval()
        tokens = initial_tokens.clone()

        with torch.no_grad():
            #* Prefill (prompt tronqué à gauche si trop long)
            past_kvs, pos, next_logits = self._prefill(tokens[:, -self.seq_length:])

            #* Decode loop (T=1 à chaque step)
            while tokens.shape[1] < max_length:
                next_token = self._sample(next_logits, temperature, top_k)
                tokens = torch.cat([tokens, next_token], dim=1)
                if tokens.shape[1] >= max_length:
                    break

                if pos >= self.seq_length:
                    #& Fenêtre pleine : re-prefill sur les derniers tokens
                    keep = max(1, self.seq_length // 2)
                    past_kvs, pos, next_logits = self._prefill(tokens[:, -keep:])
                    continue

                x = self.embedding(next_token, pos_offset=pos)  # [B, 1, D]
                pos += 1

                new_kvs = []
                for block, past_kv in zip(self.blocks, past_kvs):
                    x, kv = block(x, past_kv=past_kv, use_cache=True)
                    new_kvs.append(kv)  # type: ignore[arg-type]
                past_kvs = new_kvs

                x = self.final_ln(x)
                next_logits = self.output_head(x[:, -1, :])

        self.train(was_training)
        return tokens

    @staticmethod
    def _sample( logits: torch.Tensor, temperature: float, top_k: Optional[int]):
        """
        Applique température + top-k puis sample un token.

        Args:
            logits     : [batch_size, vocab_size]
            temperature: facteur de mise à l'échelle
            top_k      : nombre de candidats conservés

        Returns:
            torch.Tensor: token sample de forme [batch_size, 1]
        """
        logits = logits / max(temperature, 1e-8)

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k, dim=-1)
            
            #& Masquer tout ce qui est sous le k-ième score
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        probs = torch.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            return torch.argmax(logits, dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)


#* ---------------------------------------------------------------------------
#*                          Configuration helper
#* ---------------------------------------------------------------------------

class GPTConfig:
    """Configuration et paramètres du modèle GPT."""

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
            "dropout": self.dropout
        }