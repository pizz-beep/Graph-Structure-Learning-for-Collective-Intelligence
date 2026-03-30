"""
Data loading utilities
-----------------------
1. SyntheticGraphDataset  — Stochastic block model, ground truth A known
2. MetrLADataset          — 207 LA traffic sensors, ground truth = road network

METR-LA download:
  https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
  Place metr-la.h5 and adj_mx.pkl in data/metrla/
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


class SyntheticGraphDataset(Dataset):
    """
    Stochastic block model: N nodes in K communities.
    Within-community edge prob: p_in. Cross-community: p_out.
    Node features: community centroid + Gaussian noise.
    Task: node classification (predict community membership).
    """

    def __init__(
        self,
        num_nodes: int = 50,
        num_classes: int = 5,
        in_features: int = 4,
        num_samples: int = 1000,
        p_in: float = 0.7,
        p_out: float = 0.05,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)

        labels = np.array([i % num_classes for i in range(num_nodes)])

        # Ground truth adjacency (SBM)
        A_true = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                p = p_in if labels[i] == labels[j] else p_out
                if rng.random() < p:
                    A_true[i, j] = A_true[j, i] = 1.0

        centroids = rng.normal(0, 2, (num_classes, in_features))

        features = []
        for _ in range(num_samples):
            x = centroids[labels] + rng.normal(0, noise_std, (num_nodes, in_features))
            features.append(x.astype(np.float32))

        self.features = torch.from_numpy(np.stack(features))  # [S, N, F]
        self.labels = torch.from_numpy(labels).long()          # [N]
        self.A_true = torch.from_numpy(A_true).float()         # [N, N]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels


class MetrLADataset(Dataset):
    """
    METR-LA: 207 traffic sensors in Los Angeles.
    Each sample = per-sensor [mean_speed, std_speed] over a time window.
    Task: binary anomaly detection per sensor.
    Ground truth graph: road network adjacency (available for recovery eval).
    """

    def __init__(
        self,
        data_path: str = "data/metrla/metr-la.h5",
        adj_path: str = "data/metrla/adj_mx.pkl",
        window: int = 12,
        stride: int = 1,
    ):
        try:
            import pandas as pd
            import pickle

            df = pd.read_hdf(data_path)
            self.speeds = torch.tensor(df.values, dtype=torch.float32)  # [T, N]

            with open(adj_path, 'rb') as f:
                _, _, self.A_true = pickle.load(f, encoding='latin1')
            self.A_true = torch.tensor(self.A_true, dtype=torch.float32)
            print(f"[MetrLA] Loaded {self.speeds.shape[0]} timesteps, {self.speeds.shape[1]} sensors.")

        except FileNotFoundError:
            print("[MetrLA] Data files not found — using synthetic placeholder.")
            print("[MetrLA] Download: https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX")
            T, N = 1000, 207
            self.speeds = torch.randn(T, N)
            self.A_true = (torch.rand(N, N) > 0.95).float()

        self.window = window
        self.stride = stride
        T = self.speeds.shape[0]
        self.indices = list(range(0, T - window, stride))

        # Normalize per sensor
        mean = self.speeds.mean(0, keepdim=True)
        std = self.speeds.std(0, keepdim=True) + 1e-6
        self.speeds = (self.speeds - mean) / std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        window = self.speeds[t:t + self.window]           # [W, N]
        x = torch.stack([window.mean(0), window.std(0)], dim=-1)  # [N, 2]
        label = (window.abs() > 2.5).any(0).long()        # [N] binary
        return x, label


def get_dataloaders(dataset, train_ratio=0.7, val_ratio=0.1, batch_size=32, seed=42):
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )
