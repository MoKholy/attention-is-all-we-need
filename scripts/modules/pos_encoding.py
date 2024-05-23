from math import log

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embed_dim: int = 512,
        max_seq_len: int = 5000,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Seq_len x embed_dim
        encodings = torch.zeros(max_seq_len, self.embed_dim)

        # create indices, add extra dimension to broadcasts
        # Seq_len x 1
        pos = torch.arange(0, max_seq_len).unsqueeze(1)

        # embed_dim // 2
        division_term = torch.exp(
            torch.arange(0, self.embed_dim, 2) * -(log(10_000.0) / self.embed_dim)
        )

        # set encodings according to formula in paper
        encodings[:, 0::2] = torch.sin(pos * division_term)
        encodings[:, 1::2] = torch.cos(pos * division_term)

        # add batch dim
        # 1 x Seq_len x embed_dim
        pe = encodings.unsqueeze(0)
        # save tensor
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # add pos encoding to tensor, and disable grad
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


if __name__ == "__main__":

    pe = PositionalEncoding()
