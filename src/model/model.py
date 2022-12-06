import torch.nn as nn
from torch import sigmoid


class Model(nn.Module):
    """Игрушечная нейросеть для теста"""

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = sigmoid(self.fc3(x))
        return x.view(-1)
