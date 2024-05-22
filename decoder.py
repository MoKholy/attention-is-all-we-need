import torch.nn as nn

from attention import MultiHeadAttention
from embedding import Embedding
from ffn import FeedForward
from layer_norm import LayerNorm
from pos_encoding import PositionalEncoding


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        ffn_dim: int,
        attn_dropout: float,
        ffn_dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.self_attn_block = MultiHeadAttention(embed_dim, n_heads, attn_dropout)
        self.cross_attn_block = MultiHeadAttention(embed_dim, n_heads, attn_dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, ffn_dropout)
        self.layer_norm1 = LayerNorm(embed_dim, eps)
        self.layer_norm2 = LayerNorm(embed_dim, eps)
        self.layer_norm3 = LayerNorm(embed_dim, eps)

    def forward(self, x, encoder_out, input_mask, output_mask):

        input_x = x
        x = self.self_attn_block(x, x, x, output_mask)
        x = x + input_x  # residual connection 1
        x = self.layer_norm1(x)

        middle_x = x
        x = self.cross_attn_block(x, encoder_out, encoder_out, input_mask)
        x = x + middle_x  # residual connection 2
        x = self.layer_norm2(x)

        output_x = x
        x = self.ffn(x)
        x = x + output_x  # residual connection 3
        x = self.layer_norm3(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        seq_len: int,
        n_heads: int,
        ffn_dim: int,
        n_layers: int,
        attn_dropout: float,
        ffn_dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.output_embedding = Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, seq_len)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim, n_heads, ffn_dim, attn_dropout, ffn_dropout, eps
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = LayerNorm(embed_dim, eps)

    def forward(self, x, encoder_out, input_mask, output_mask):
        x = self.output_embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_out, input_mask, output_mask)
        x = self.final_norm(x)
        return x
