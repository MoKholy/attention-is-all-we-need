from dataclasses import dataclass


@dataclass
class TrainingConfigs:
    label_smoothing: float = 0.1
    n_training_steps: int = 1e5
    n_warmup: int = 4e3
