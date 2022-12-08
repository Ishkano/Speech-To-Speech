import os
import numpy as np
from dataclasses import dataclass

from librosa import load
from librosa.feature import melspectrogram

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


@dataclass(frozen=False)
class PreprocParams:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_size: float = 0.2
    train_size: float = 1 - test_size
    sample_rate: int = 48000
    random_state: int = 42
    batch_size: int = 5


class AudioSet(Dataset):

    def __init__(self, files, params):
        self.params = params
        self.data = torch.Tensor(self.data_from_files(files)).to(self.params.device)

    def data_from_files(self, files):
        out = []
        for file in files:
            sig = np.array(load(file, sr=self.params.sample_rate)[0])
            trshld = self.params.sample_rate // 2
            while len(sig) >= trshld:
                out.append(sig[:trshld])
                sig = sig[trshld:]
        return np.array(out)

    def specs_from_files(self, files):
        out = []
        for file in files:
            melspecs = melspectrogram(y=np.array(load(file, sr=self.params.sample_rate)[0]),
                                      sr=self.params.sample_rate)
            trshld = self.params.sample_rate // 2
            while len(melspecs) >= trshld:
                out.append(melspecs[:trshld])
                melspecs = melspecs[trshld:]
        return np.array(out)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Preprocessing:

    def __init__(self, params):
        self.params = params

        proj_root_path = '\\'.join(os.getcwd().split('\\')[:-2])
        data_path = '{}\\data\\dev-clean'.format(proj_root_path)

        train_files, test_files = train_test_split(self.search_for_flac(data_path),
                                                   test_size=self.params.test_size,
                                                   random_state=self.params.random_state)

        self.train_loader = self.create_loader(train_files)
        self.test_loader = self.create_loader(test_files)

    @staticmethod
    def search_for_flac(root_path):
        files = []
        for (path, _, filenames) in os.walk(root_path):
            for filename in filenames:
                if filename.split('.')[-1] == 'flac':
                    files.extend(['{}\\{}'.format(path, filename)])
        return files

    def create_loader(self, files):
        return DataLoader(AudioSet(files, self.params), batch_size=self.params.batch_size)

    def train_test_loaders(self):
        return self.train_loader, self.test_loader


if __name__ == "__main__":
    preproc = Preprocessing(PreprocParams)
    train_loader, test_loader = preproc.train_test_loaders()
    for batch, tmp in enumerate(train_loader):
        print(batch, tmp)
        if batch == 3:
            break
