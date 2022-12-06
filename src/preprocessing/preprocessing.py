import os
from dataclasses import dataclass

from torch.cuda import is_available
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PreprocParams:

    device: str = 'cuda' if is_available() else 'cpu'
    train_size: float = 0.8
    batch_size: int = 5
    epochs: int = 15
    learning_rate: int = 1e-3


class AudioSet(Dataset):

    def __init__(self, data, device):
        self.data = torch.Tensor(data).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return item


class Preprocessing:
    """
    The idea is to create data loaders for training and evaluation.
    Otherwise, we will get a OutOfMemoryError
    """

    def __init__(self, params):
        self.params = params

        self.root_path = '\\'.join(os.getcwd().split('\\')[:-2])
        self.data_path = '{}\\data\\dev-clean'.format(self.root_path)

        file_names = []
        for (path, _, filenames) in os.walk(self.data_path):
            file_names.extend(['{}\\{}'.format(path, filename) for filename in filenames])

        train, test = train_test_split(file_names, train_size=self.params.train_size, random_state=42)

        self.train_loader = self.create_loader(train)
        self.test_loader = self.create_loader(test)

    def create_loader(self, data):
        return DataLoader(AudioSet(data, self.params.device), batch_size=self.params.batch_size)


if __name__ == "__main__":
    preproc = Preprocessing(PreprocParams)
