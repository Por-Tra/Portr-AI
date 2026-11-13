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

File name : gpt_model.py

description du script:
    
    Implémentation complète du LLM style GPT.

    Architecture :

        1. Tokens [batch, seq]
        
        2. Embedding (Token + Position + Dropout) [batch, seq, embedding_dim]
        
        3. N * TransformerBlock (Attention + MLP + Résiduel) [batch, seq, embedding_dim]
        
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

    def resize_token_embeddings(self, new_vocab_size):
        """Agrandit l'embedding de tokens en conservant les poids existants.

        Les nouvelles lignes sont initialisees comme l'embedding GPT-2, puis
        la tete de sortie est recreee et liee au nouvel embedding.

        Args:
            new_vocab_size (int): Nouvelle taille du vocabulaire (doit être >= vocab_size actuel).

        Raises:
            ValueError: Si la nouvelle taille du vocabulaire est inférieure à la taille actuelle.
        """
        if new_vocab_size < self.vocab_size:
            raise ValueError(f"Le nouveau vocabulaire ({new_vocab_size}) ne peut pas etre plus petit que l'ancien ({self.vocab_size}).")
        
        if new_vocab_size == self.vocab_size:
            return

        old_token_emb = self.embedding.token_emb
        new_token_emb = nn.Embedding(new_vocab_size, self.embedding_dim, device=old_token_emb.weight.device, dtype=old_token_emb.weight.dtype)
        nn.init.normal_(new_token_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_token_emb.weight[: self.vocab_size].copy_(old_token_emb.weight)

        self.embedding.token_emb = new_token_emb
        self.embedding.vocab_size = new_vocab_size
        self.vocab_size = new_vocab_size

        self.output_head = nn.Linear(
            self.embedding_dim,
            new_vocab_size,
            bias=False,
            device=new_token_emb.weight.device,
            dtype=new_token_emb.weight.dtype
        )
        self.output_head.weight = self.embedding.token_emb.weight

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

    def _prefill(self, input_tokens):
        """Encode une séquence complète et construit le KV cache.

        Args:
            input_tokens (torch.Tensor): [batch_size, seq_len] - indices des tokens d'entrée

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

    def generate(self, initial_tokens, max_length, temperature = 1.0, top_k = None, eos_token_id = None):
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
            eos_token_id  : Arrête la génération quand ce token est produit.

        Raises:
            ValueError: Si initial_tokens n'est pas [B, T] non vide.
            ValueError: Si max_length n'est pas un entier positif.
            ValueError: Si temperature n'est pas finie ou < 0.
            ValueError: Si top_k n'est pas un entier positif ou None.
            ValueError: Si eos_token_id n'est pas un entier entre 0 et vocab_size-1.

        Returns:
            torch.Tensor: tokens générés de forme [batch_size, max(max_length, prompt_len)]
        """
        if initial_tokens.ndim != 2 or initial_tokens.shape[0] == 0 or initial_tokens.shape[1] == 0:
            raise ValueError("initial_tokens doit etre [B,T] non vide")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 0:
            raise ValueError("max_length doit etre un entier >= 0")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature doit etre finie et >= 0 (0 = glouton)")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("top_k doit etre un entier positif ou None")
        if eos_token_id is not None and not 0 <= eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id hors vocabulaire")
        tokens = initial_tokens.clone()
        if tokens.shape[1] >= max_length:
            return tokens
        
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                past_kvs, pos, next_logits = self._prefill(tokens[:, -self.seq_length:])
                finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
                while tokens.shape[1] < max_length:
                    next_token = self._sample(next_logits, temperature, top_k)

                    if eos_token_id is not None:
                        next_token = torch.where(finished[:, None], eos_token_id, next_token)
                        finished |= next_token[:, 0] == eos_token_id

                    tokens = torch.cat([tokens, next_token], dim=1)
                    if eos_token_id is not None and bool(finished.all()):
                        break
                    if tokens.shape[1] >= max_length:
                        break
                    if pos >= self.seq_length:
                        keep = max(1, self.seq_length // 2)
                        past_kvs, pos, next_logits = self._prefill(tokens[:, -keep:])
                        continue
                    x = self.embedding(next_token, pos_offset=pos)
                    pos += 1
                    new_kvs = []
                    for block, past_kv in zip(self.blocks, past_kvs):
                        x, kv = block(x, past_kv=past_kv, use_cache=True)
                        new_kvs.append(kv)
                    past_kvs = new_kvs
                    next_logits = self.output_head(self.final_ln(x)[:, -1, :])
            return tokens
        finally:
            self.train(was_training)


    @staticmethod
    def _sample( logits, temperature, top_k):
        """
        Applique température + top-k puis sample un token.

        Args:
            logits     : [batch_size, vocab_size]
            temperature: facteur de mise à l'échelle
            top_k      : nombre de candidats conservés

        Raises:
            ValueError: Si temperature n'est pas finie ou < 0.
            ValueError: Si top_k n'est pas un entier positif ou None.
            ValueError: Si eos_token_id n'est pas un entier entre 0 et vocab_size-1.
            ValueError: Si les logits contiennent des valeurs non finies (NaN ou Inf).

        Returns:
            torch.Tensor: token sample de forme [batch_size, 1]
        """
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature invalide")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise ValueError("top_k invalide")
        if not torch.isfinite(logits).all():
            raise ValueError("Logits non finis pendant la generation")
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k, dim=-1)
            
            #& Masquer tout ce qui est sous le k-ième score
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        probs = torch.softmax(logits, dim=-1)

        if not torch.isfinite(probs).all():
            raise ValueError("Probabilites non finies pendant la generation")

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

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "seq_length": self.seq_length,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
            "dropout": self.dropout
        }