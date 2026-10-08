import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.size(-1)
    
    # Compute attention scores: (seq_len, d_k) x (d_k, seq_len) -> (seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply softmax along the last dimension to get weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values: (seq_len, seq_len) x (seq_len, d_k) -> (seq_len, d_k)
    output = torch.matmul(attn_weights, V)
    return output

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    d_k = d_model // n_heads

    Q_heads = Q.view(seq_len, n_heads, d_k).transpose(0, 1)
    K_heads = K.view(seq_len, n_heads, d_k).transpose(0, 1)
    V_heads = V.view(seq_len, n_heads, d_k).transpose(0, 1)

    scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)

    # (n_heads, seq_len, seq_len) x (n_heads, seq_len, d_k) -> (n_heads, seq_len, d_k)
    head_outputs = torch.matmul(attn_weights, V_heads)

    # Transpose back and concatenate heads: (seq_len, n_heads, d_k) -> (seq_len, d_model)
    concat_output = head_outputs.transpose(0, 1).contiguous().view(seq_len, d_model)

    return concat_output