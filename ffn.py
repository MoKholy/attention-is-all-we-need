import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


if __name__ == "__main__":
    ffn = FeedForward(5, 3, 0.5)
    x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    print(f"{ffn(x).shape = }")
