import numpy as np
import os
from torch.utils.data import Dataset
import multiFileDataset


class multiFileDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

        # Load metadata from each file
        self.file_sizes = []
        for f in file_paths:
            data = np.load(f)
            self.file_sizes.append(len(data["X"]))

        self.total_size = sum(self.file_sizes)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Determine which file this index falls in
        file_idx = 0
        while idx >= self.file_sizes[file_idx]:
            idx -= self.file_sizes[file_idx]
            file_idx += 1

        data = np.load(self.file_paths[file_idx])

        X = data["X"][idx]
        E = data["Y_E"][idx]
        PSI = data["Y_PSI"][idx]

        return X.astype(np.float32), E.astype(np.float32), PSI.astype(np.float32)
