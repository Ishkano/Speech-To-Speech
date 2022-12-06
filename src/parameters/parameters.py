from dataclasses import dataclass


@dataclass
class Parameters:

    model_type: str = 'default'
    loss_fn: str = 'default'
    optimizer: str = 'default'

    epochs: int = 10
    batch_size: int = 12
    learning_rate: float = 1e-3
