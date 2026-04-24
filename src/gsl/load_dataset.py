import pickle
import os
from pathlib import Path

# Fix paths so they work from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADJ_PATH = PROJECT_ROOT / "data" / "adj_METR-LA.pkl"
H5_PATH = PROJECT_ROOT / "data" / "METR-LA.h5"

with open(ADJ_PATH, "rb") as f:
    sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding="latin1")

print("Number of sensors:", len(sensor_ids))
print("Adjacency shape:", adj_mx.shape)
print("Sample sensor IDs:", sensor_ids[:5])
print("Adjacency sample:\n", adj_mx[:5, :5])

import pandas as pd

df = pd.read_hdf(H5_PATH)

print("\n--- METR-LA DATASET ---")
print(f"Shape: {df.shape[0]} time steps, {df.shape[1]} sensors")
print(df) # This will print the dataframe showing the first few and last few rows, which is best for viewing large datasets

import h5py

# Get underlying data array from the dataframe for visualization
data = df.values.astype(float)
print("Underlying Numpy Array Shape:", data.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.imshow(data[:100], aspect='auto', cmap='viridis')
plt.colorbar(label="Speed (mph)")
plt.title("Traffic Speeds Over Time (First 100 timesteps)")
plt.xlabel("Sensors")
plt.ylabel("Time Step")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(adj_mx, cmap='hot', aspect='auto')
plt.title("Adjacency Matrix (Road Connections)")
plt.colorbar()
plt.tight_layout()
plt.show()