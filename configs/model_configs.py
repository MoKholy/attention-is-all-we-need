from dataclasses import dataclass


@dataclass
class ModelConfig:
    embed_dim: int = 512
    n_heads: int = 8
    ffn_dim: int = 2048
    n_layers: int = 12
    attn_dropout: float = 0.15
    ffn_dropout: float = 0.15
    pe_dropout: float = 0.15
    eps: float = 1e-6
