from math import sqrt

import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        assert (
            embed_dim % n_heads == 0
        ), "Embedding dimension must be divisible by number of Attention Heads"

        self.head_dim = self.embed_dim // self.n_heads

        self.qw = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.vw = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.kw = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.ow = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        v,
        mask,
    ):
        q = self.qw(q)  # B x seq_len x embed_dim
        k = self.kw(k)  # B x seq_len x embed_dim
        v = self.vw(v)  # B x seq_len x embed_dim

        batch_size, seq_len, _ = q.shape
        # change shape for multihead attentions
        q = q.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        )  # B x seq_len x n_heads x head_dim
        k = k.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        )  # B x seq_len x n_heads x head_dim
        v = v.view(
            batch_size, seq_len, self.n_heads, self.head_dim
        )  # B x seq_len x n_heads x head_dim

        # transpose, so each head sees seq_len x head_dim
        # B x n_heads x seq_len x head_dim
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute attention
        x, self.attn_scores = MultiHeadAttention.attention(q, k, v, mask, self.dropout)
        # B x n_heads x seq_len x head_dim -> B x seq_len x n_heads x head_dim -> B x seq_len x embed_dim
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.n_heads * self.head_dim)
        )

        # B x seq_len x embed_dim -> B x seq_len x embed_dim
        x = self.ow(x)
        return x

    @staticmethod
    def attention(q, k, v, mask, dropout):

        _, _, _, head_dim = q.shape
        # compute attention as in paper
        # k transposed from B x n_heads x seq_len x head_dim to B x n_heads x head_dim x seq_len, this gives
        # B x n_heads x seq_len x seq_len after matmul which is needed later when we multiply by V
        attn_scores = (q @ k.transpose(2, 3)) / sqrt(head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill_(mask == False, value=float("-inf"))

        # apply softmax as done in paper
        attn_scores = F.softmax(attn_scores, dim=-1)

        if dropout:
            attn_scores = dropout(attn_scores)

        # x: B x n_heads x seq_len x head_dim
        x = attn_scores @ v
        return x, attn_scores
