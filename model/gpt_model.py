"""
gpt_model.py

Implémentation complète du LLM en style GPT.

Architecture complète :

Tokens [batch, seq]
↓
Embedding
  - Token embedding
  - Position embedding
  - LayerNorm + Dropout
[batch, seq, embedding_dim]
↓
N TransformerBlocks
  - Self-Attention + Residual + LayerNorm
  - FeedForward + Residual + LayerNorm
[batch, seq, embedding_dim]
↓
Final LayerNorm
[batch, seq, embedding_dim]
↓
Output Projection (vocab_size)
[batch, seq, vocab_size]
↓
Logits (convertis en probabilités par softmax pendant l'inférence)
"""

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
    
    C'est l'assemblage complet de tous les composants.
    """
    
    def __init__(self, vocab_size: int, seq_length: int, embedding_dim: int = 256,
        num_heads: int = 8, num_blocks: int = 4, dropout: float = 0.1):
        """
        Args:
            vocab_size: Nombre de tokens uniques dans le vocabulaire
            seq_length: Longueur maximale des séquences
            embedding_dim: Dimension des embeddings et hidden states
            num_heads: Nombre de têtes d'attention dans chaque bloc
            num_blocks: Nombre de blocs Transformer à empiler
            dropout: Taux de dropout
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embedding = Embedding(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=embedding_dim,
        )
        
        # Stack de Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim=embedding_dim,
                num_heads=num_heads, dropout=dropout,)
            for _ in range(num_blocks) # On empile num_blocks blocs Transformer
        ])
        
        # Final layer normalization
        self.final_ln = nn.LayerNorm(embedding_dim)
        
        # Output projection : transforme hidden state en logits over vocabulary
        self.output_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Dropout final
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du modèle.
        
        Args:
            tokens: [batch_size, seq_length] - indices des tokens
        
        Returns:
            [batch_size, seq_length, vocab_size] - logits pour chaque position
        """
        # Embedding
        x = self.embedding(tokens)  # [batch_size, seq_length, embedding_dim]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)  # [batch_size, seq_length, embedding_dim]
        
        # Final layer norm
        x = self.final_ln(x)  # [batch_size, seq_length, embedding_dim]
        
        # Output projection
        logits = self.output_head(x)  # [batch_size, seq_length, vocab_size]
        
        return logits
    
    def generate(self, initial_tokens: torch.Tensor, max_length: int,
        temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Génère des tokens en continuant à partir d'une séquence initiale.
        
        Args:
            initial_tokens: [batch_size, seq_length] - séquence initiale
            max_length: Nombre total de tokens à générer
            temperature: Contrôle la diversité (1.0 = normal, < 1.0 = moins diverse)
            top_k: Si spécifié, sample parmi les top-k tokens les plus probables
        
        Returns:
            [batch_size, max_length] - séquence générée
        """
        current_tokens = initial_tokens.clone()
        
        with torch.no_grad():
            while current_tokens.shape[1] < max_length:
                # Prendre seulement les tokens dans la fenêtre de contexte
                if current_tokens.shape[1] > self.seq_length:
                    input_tokens = current_tokens[:, -self.seq_length:]
                else:
                    input_tokens = current_tokens
                
                # Forward pass
                logits = self.forward(input_tokens)  # [batch_size, seq_length, vocab_size]
                
                # Récupérer les logits du dernier token
                next_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Appliquer la température
                next_logits = next_logits / temperature
                
                # Top-k sampling (optionnel)
                if top_k is not None:
                    values, indices = torch.topk(next_logits, min(top_k, self.vocab_size), dim=-1)
                    next_logits_filtered = torch.full_like(next_logits, -float('inf'))
                    next_logits_filtered.scatter_(-1, indices, values)
                    next_logits = next_logits_filtered
                
                # Softmax pour obtenir les probabilités
                probs = torch.softmax(next_logits, dim=-1)
                
                # Vérifier qu'il n'y a pas de NaN
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    # Fallback
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    # Sample le prochain token
                    next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
                
                # Ajouter à la séquence
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        return current_tokens


# Petite classe utile pour charger facilement le modèle
class GPTConfig:
    """Configuration du modèle GPT."""
    
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get('vocab_size', 10000)
        self.seq_length = kwargs.get('seq_length', 512)
        self.embedding_dim = kwargs.get('embedding_dim', 256)
        self.num_heads = kwargs.get('num_heads', 8)
        self.num_blocks = kwargs.get('num_blocks', 4)
        self.dropout = kwargs.get('dropout', 0.1)
    
    def to_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'seq_length': self.seq_length,
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            'num_blocks': self.num_blocks,
            'dropout': self.dropout,
        }
