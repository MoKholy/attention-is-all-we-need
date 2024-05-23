from math import sqrt

import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()

        self.input_dim = vocab_size
        self.embed_dim = embed_dim
        # vocab_size x embed_dim
        self.embed = nn.Embedding(vocab_size, self.embed_dim)

    def forward(self, x) -> torch.Tensor:

        # scale by sqrt of embed_dim, from paper
        # B x Seq_len -> B x Vocab_size x embed_dim
        x = self.embed(x) * sqrt(self.embed_dim)
        return x


if __name__ == "__main__":
    test = Embedding(10, 3)
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    print(f"{test(input).shape = }")
