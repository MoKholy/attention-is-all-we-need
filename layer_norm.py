import torch.nn as nn
import torch


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps  # small number to prevent division by zero errors
        self.alpha = nn.Parameter(torch.ones(dim))  # multiply
        self.beta = nn.Parameter(torch.zeros(dim))  # add

    def forward(self, x):
        # get mean and std for each sentence
        mu = x.mean(dim=-1, keep_dim=True)
        std = x.std(dim=-1, keed_dim=True)

        x = self.alpha * ((x - mu) / (std + self.eps)) + self.beta
        return x
