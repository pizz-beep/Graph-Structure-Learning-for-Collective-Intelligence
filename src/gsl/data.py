# src/gsl/data.py

import numpy as np
import pandas as pd
import pickle
import urllib.request
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# src/gsl/data.py

# ─────────────────────────────────────────────
# Project paths
# ─────────────────────────────────────────────
# src/gsl/data.py -> go 2 levels up -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Actual filenames on disk — keep these in sync with whatever is in data/
H5_PATH  = DATA_DIR / "METR-LA.h5"       # was erroneously metr-la.h5
PKL_PATH = DATA_DIR / "adj_METR-LA.pkl"  # was erroneously adj_mx.pkl

# ─────────────────────────────────────────────
# 1. Raw loading
# ─────────────────────────────────────────────

def load_metr_la():
    """
    Loads metr-la.h5 directly from project_root/data/
    """
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Missing dataset file: {H5_PATH}")

    df = pd.read_hdf(H5_PATH)
    data = df.values.astype(np.float32)

    print(f"Loaded METR-LA: {data.shape[0]} timesteps, {data.shape[1]} sensors")
    return data, df.columns.tolist()


def download_adj_matrix(save_dir):
    """
    Downloads the official METR-LA road adjacency matrix (adj_mx.pkl)
    from the DCRNN repo if not already present.
    Returns the file path.
    """
    url = (
        "https://raw.githubusercontent.com/liyaguang/DCRNN/"
        "master/data/sensor_graph/adj_mx.pkl"
    )
    save_path = os.path.join(save_dir, "adj_mx.pkl")
    if not os.path.exists(save_path):
        print("Downloading adj_mx.pkl ...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Saved to {save_path}")
    else:
        print("adj_mx.pkl already exists, skipping download.")
    return save_path


def load_adj_matrix():
    """
    Loads adj_mx.pkl directly from project_root/data/
    """
    if not PKL_PATH.exists():
        raise FileNotFoundError(f"Missing adjacency file: {PKL_PATH}")

    with open(PKL_PATH, "rb") as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(
            f, encoding="latin1"
        )

    adj_mx = adj_mx.astype(np.float32)
    print(f"Loaded adjacency matrix: {adj_mx.shape}")

    return adj_mx, sensor_ids, sensor_id_to_ind


# ─────────────────────────────────────────────
# 2. Normalization
# ─────────────────────────────────────────────

class StandardScaler:
    """
    Z-score normalization fitted on training data only.
    We store mean and std so we can invert the transform when computing
    real-world metrics (MAE in miles/hour, not in normalized units).
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        # data shape: (T, N) — fit across all timesteps and sensors
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data):
        return data * self.std + self.mean


# ─────────────────────────────────────────────
# 3. Train / val / test split
# ─────────────────────────────────────────────

def split_data(data, train_ratio=0.7, val_ratio=0.1):
    """
    Splits (T, N) array chronologically — no shuffling.
    Shuffling time series data leaks future into the past.
    Returns three (T_split, N) arrays.
    """
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    train = data[:train_end]
    val   = data[train_end:val_end]
    test  = data[val_end:]

    print(f"Split → train: {train.shape}, val: {val.shape}, test: {test.shape}")
    return train, val, test


# ─────────────────────────────────────────────
# 4. PyTorch Dataset
# ─────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Turns a (T, N) array into overlapping windows for sequence prediction.

    Each sample is:
        x : (in_steps, N)  — the input window (what each sensor saw)
        y : (out_steps, N) — the target window (what we want to predict)

    Example with in_steps=12, out_steps=3:
        x = speeds at t=0..11 across all 207 sensors
        y = speeds at t=12..14 across all 207 sensors

    The window slides forward by 1 timestep each sample.
    """
    def __init__(self, data, in_steps=12, out_steps=3):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.in_steps = in_steps
        self.out_steps = out_steps

    def __len__(self):
        return len(self.data) - self.in_steps - self.out_steps + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.in_steps]                          # (in_steps, N)
        y = self.data[idx + self.in_steps : idx + self.in_steps + self.out_steps]  # (out_steps, N)
        return x, y


# ─────────────────────────────────────────────
# 5. Convenience builder
# ─────────────────────────────────────────────

def build_dataloaders(h5_path, adj_dir, in_steps=12, out_steps=3,
                      batch_size=32, train_ratio=0.7, val_ratio=0.1):
    """
    One call that does everything:
      - loads the raw sensor data
      - downloads the road adjacency if needed
      - normalizes (fit on train only)
      - splits chronologically
      - wraps in DataLoaders

    Returns:
        train_loader, val_loader, test_loader,
        scaler (to invert predictions back to mph),
        adj_mx (207x207 numpy array, ground truth graph)
    """
    # Load raw data — honour the caller-supplied path, fall back to constant
    if h5_path is not None:
        _h5 = Path(h5_path)
        if not _h5.exists():
            raise FileNotFoundError(f"h5 file not found: {_h5}")
        df = pd.read_hdf(_h5)
        data = df.values.astype(np.float32)
    else:
        data, _ = load_metr_la()

    # Download + load ground truth adjacency
    if adj_dir is not None:
        adj_path = download_adj_matrix(adj_dir)
        adj_mx, _, _ = load_adj_matrix()  # still reads from PKL_PATH
    else:
        adj_mx, _, _ = load_adj_matrix()

    # Normalize
    scaler = StandardScaler()
    train_raw, val_raw, test_raw = split_data(data, train_ratio, val_ratio)
    scaler.fit(train_raw)                  # fit ONLY on train

    train_norm = scaler.transform(train_raw)
    val_norm   = scaler.transform(val_raw)
    test_norm  = scaler.transform(test_raw)

    # Datasets
    train_ds = SlidingWindowDataset(train_norm, in_steps, out_steps)
    val_ds   = SlidingWindowDataset(val_norm,   in_steps, out_steps)
    test_ds  = SlidingWindowDataset(test_norm,  in_steps, out_steps)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"\nBatches → train: {len(train_loader)}, "
          f"val: {len(val_loader)}, test: {len(test_loader)}")

    return train_loader, val_loader, test_loader, scaler, adj_mx


# ─────────────────────────────────────────────
# 6. Synthetic dataset for ablation / sanity check
# ─────────────────────────────────────────────

class SyntheticGraphDataset(Dataset):
    """
    Generates a synthetic graph classification/regression dataset.

    Process:
        1. Sample a random Erdős–Rényi graph with edge probability p.
           This becomes A_true — the ground truth we try to recover.
        2. Generate node features by diffusing Gaussian noise over the
           true graph (so nearby nodes have correlated features).
        3. Label each node by the sum of its neighbors' raw features,
           discretized into num_classes bins.

    Why this structure?
        If the GSL module recovers A_true, it can trivially solve the task.
        If it fails to recover the graph, task performance drops.
        This tight coupling makes graph recovery and task success co-vary —
        which is exactly the relationship your experiments want to show.

    Attributes:
        features : (num_samples, num_nodes, in_features) — float32 tensor
        labels   : (num_samples, num_nodes)              — int64 tensor
        A_true   : (num_nodes, num_nodes)                — float32 tensor
    """

    def __init__(
        self,
        num_nodes: int = 50,
        in_features: int = 4,
        num_samples: int = 500,
        num_classes: int = 5,
        edge_prob: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        rng = np.random.default_rng(seed)

        # ── Ground-truth graph (Erdős–Rényi) ─────────────────────────────
        adj = (rng.random((num_nodes, num_nodes)) < edge_prob).astype(np.float32)
        np.fill_diagonal(adj, 0)                  # no self-loops
        adj = np.maximum(adj, adj.T)              # make symmetric
        # Row-normalize so message-passing sums to 1
        deg = adj.sum(axis=1, keepdims=True).clip(min=1)
        adj_norm = adj / deg
        self.A_true = torch.tensor(adj_norm, dtype=torch.float32)

        # ── Node features: diffused Gaussian noise ────────────────────────
        # Each raw feature is IID Gaussian; one step of graph diffusion
        # makes neighboring nodes correlated — the model can exploit this.
        raw = rng.standard_normal((num_samples, num_nodes, in_features)).astype(np.float32)
        # One diffusion step: h = A_norm @ x
        diffused = adj_norm @ raw                 # (N, N) x (S, N, F) via broadcast
        features = 0.5 * raw + 0.5 * diffused     # (num_samples, N, F)

        # ── Labels: binned neighbor-sum ───────────────────────────────────
        # neighbor_sum[s, i] = sum of features of node i's true neighbors
        neighbor_sum = (adj_norm @ features).sum(axis=-1)  # (S, N)
        # Bin into num_classes equal-width intervals
        lo, hi = neighbor_sum.min(), neighbor_sum.max()
        bins = np.linspace(lo, hi, num_classes + 1)
        labels = np.digitize(neighbor_sum, bins[1:-1]).astype(np.int64)  # (S, N)

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)
        self.num_nodes   = num_nodes
        self.in_features = in_features
        self.num_classes = num_classes

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def get_dataloaders(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
):
    """
    Splits any Dataset into train / val / test DataLoaders.

    Uses a reproducible random split (not chronological, since synthetic
    data has no time ordering). For time-series datasets use split_data()
    and SlidingWindowDataset instead.

    Returns:
        train_loader, val_loader, test_loader
    """
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Synthetic split -> train: {n_train}, val: {n_val}, test: {n_test}")
    return train_loader, val_loader, test_loader