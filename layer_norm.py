import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps  # small number to prevent division by zero errors
        self.alpha = nn.Parameter(torch.ones(dim))  # multiply
        self.beta = nn.Parameter(torch.zeros(dim))  # add

    def forward(self, x):
        # get mean and std for each sent
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = self.alpha * ((x - mu) / (std + self.eps)) + self.beta
        return x
