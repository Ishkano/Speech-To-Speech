import torch
from dataclasses import dataclass


@dataclass(frozen=False)
class Parameters:

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_size: float = 0.2
    train_size: float = 1 - test_size

    model_type: str = 'default'
    loss_fn: str = 'default'
    optimizer: str = 'default'

    epochs: int = 10
    batch_size: int = 12
    learning_rate: float = 1e-3
