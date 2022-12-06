import torch
from dataclasses import dataclass


@dataclass
class Parameters:

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_size: float = 0.8

    model_type: str = 'default'
    loss_fn: str = 'default'
    optimizer: str = 'default'

    epochs: int = 10
    batch_size: int = 12
    learning_rate: float = 1e-3
