"""
transformer_block.py

Structure complète d’un bloc
x
↓
Multi-Head Attention
↓
Add & Norm
↓
FeedForward
↓
Add & Norm
↓
sortie


    
"""



import torch
import torch.nn as nn
try:
    from .attention import Attention
except ImportError:
    from attention import Attention


class MLP(nn.Module):
    """
    Réseau de neurones classique appliqué à chaque token indépendamment.
    
    Formule :
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    En pratique :
    Linear -> Activation (GELU) -> Linear -> Dropout
    
    Intuition :
    - Attention = communication entre tokens
    - FeedForward = réflexion individuelle du token
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, embedding_dim]
        Returns:
            [batch_size, seq_length, embedding_dim]
        """
        x = self.fc1(x)           # Linear expansion
        x = self.gelu(x)          # Activation
        x = self.dropout(x)       # Dropout
        x = self.fc2(x)           # Linear projection back
        x = self.dropout(x)       # Dropout
        return x


class TransformerBlock(nn.Module):
    """
    Bloc Transformer complet : Attention + FeedForward
    avec résiduel et normalisation.
    
    Architecture (pre-norm style) :
    - LayerNorm avant Attention
    - Connexion résiduelle après
    - LayerNorm avant FeedForward
    - Connexion résiduelle après
    """
    
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Attention block
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = Attention(embedding_dim, num_heads, dropout)
        
        # FeedForward block
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, 4 * embedding_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, embedding_dim]
        Returns:
            [batch_size, seq_length, embedding_dim]
        """
        # Pre-norm style (normalize before, not after)
        # Self-Attention block with residual
        x = x + self.attention(self.ln1(x))
        
        # FeedForward block with residual
        x = x + self.mlp(self.ln2(x))
        
        return x

    