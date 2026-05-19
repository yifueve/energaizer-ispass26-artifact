import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..utils import log_dict


FEATURE_COLUMNS = [
    "vmcategory_numeric",
    "vmcorecountbucket",
    "vmmemorybucket",
    "lifetime",
    "corehour",
]
TARGET_COLUMN = "avgcpu"

FEATURE_MEAN = torch.tensor(
    [1.929186, 3.166781, 12.205683, 30.259999, 12.340768],
    dtype=torch.float32,
)
FEATURE_STD = torch.tensor(
    [0.346755, 3.397199, 12.914447, 133.498923, 171.776218],
    dtype=torch.float32,
)


class DatacenterDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_ids, train, split_ratio=0.8, seed=0):
        self.data_dir = data_dir
        self.file_ids = list(file_ids)
        self.train = train
        self.split_ratio = split_ratio
        self.seed = seed

        frames = []
        for file_id in self.file_ids:
            path = os.path.join(data_dir, f"datacenter_{file_id}.csv")
            data = pd.read_csv(path, usecols=FEATURE_COLUMNS + [TARGET_COLUMN])
            indices = self._split_indices(len(data), file_id)
            frames.append(data.iloc[indices])

        data = pd.concat(frames, ignore_index=True)
        features = data[FEATURE_COLUMNS].astype(np.float32).values
        targets = data[TARGET_COLUMN].astype(np.float32).values

        self.features = torch.from_numpy(features)
        self.features = (self.features - FEATURE_MEAN) / FEATURE_STD
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def _split_indices(self, n, file_id):
        rng = np.random.RandomState(self.seed + file_id)
        indices = rng.permutation(n)
        split = int(n * self.split_ratio)
        if self.train:
            return indices[:split]
        return indices[split:]


class DatacenterMLP(nn.Module):
    def __init__(self, input_dim=len(FEATURE_COLUMNS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_datacenter_model():
    return DatacenterMLP(input_dim=len(FEATURE_COLUMNS))


def datacenter(
    data_dir,
    file_ids,
    train,
    batch_size,
    shuffle=None,
    sampler_callback=None,
    split_ratio=0.8,
    seed=0,
    drop_last=True,
    **loader_kwargs,
):
    if sampler_callback is not None and shuffle is not None:
        raise ValueError

    dataset = DatacenterDataset(
        data_dir=data_dir,
        file_ids=file_ids,
        train=train,
        split_ratio=split_ratio,
        seed=seed,
    )
    sampler = sampler_callback(dataset) if sampler_callback else None
    log_dict(
        {
            "Type": "Setup",
            "Dataset": "datacenter",
            "data_dir": data_dir,
            "file_ids": list(file_ids),
            "train": train,
            "split_ratio": split_ratio,
            "seed": seed,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler.__str__() if sampler else None,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
        }
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )
