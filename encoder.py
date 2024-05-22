import torch.nn as nn
import torch.nn.functional as F
import torch
from attention import MultiHeadAttention
from embedding import Embedding
from ffn import FeedForward
from layer_norm import LayerNorm
from pos_encoding import PositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ffn_dim: int,
        attn_dropout: float,
        ffn_dropout: float,
        eps: float,
    ) -> None:
        self.multi_head_attn = MultiHeadAttention(embed_dim, n_heads, attn_dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, ffn_dropout)
        self.layer_norm1 = LayerNorm(embed_dim, eps)
        self.layer_norm2 = LayerNorm(embed_dim, eps)

    def forward(self, x, mask):

        # pass to attention block
        input_x = x
        x = self.multi_head_attn(x, x, x, mask)
        x = input_x + x  # residual connection 1
        x = self.layer_norm1(x)

        middle_x = x
        x = self.ffn(x)
        x = middle_x + x  # residual connection 2
        x = self.layer_norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        attn_dropout: float,
        ffn_dropout: float,
        eps: float,
    ) -> None:

        # create embedding
        self.input_embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max)
