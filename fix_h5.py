"""
fix_h5.py — Re-saves METR-LA.h5 in a pandas-compatible format.

The original file was written by an older version of PyTables/pandas that
stored the group-type as bytes. Newer pandas (>= 2.x) expects a str, causing:
    TypeError: a bytes-like object is required, not 'str'

Fix: read the raw matrix with h5py (no pandas metadata involved), then
write a fresh HDF5 file via pandas so the metadata is correct.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
H5_PATH  = DATA_DIR / "METR-LA.h5"
BAK_PATH = DATA_DIR / "METR-LA.h5.bak"
TMP_PATH = DATA_DIR / "METR-LA_fixed.h5"

def fix_with_h5py():
    """Try reading via h5py (no pandas) then re-save."""
    try:
        import h5py
    except ImportError:
        print("h5py not installed — run:  pip install h5py")
        return False

    print(f"Opening {H5_PATH} with h5py …")
    with h5py.File(H5_PATH, "r") as f:
        # METR-LA stores data under key 'df' or root-level dataset
        print("  Keys:", list(f.keys()))
        key = list(f.keys())[0]
        grp = f[key]
        print("  Sub-keys:", list(grp.keys()) if hasattr(grp, "keys") else "dataset")

        # Try to reconstruct a DataFrame
        # Typical PyTables fixed-format layout:
        #   /df/block0_values  (float32 matrix  T x N)
        #   /df/axis0          (column labels  N)
        #   /df/axis1          (row index      T)
        if "block0_values" in grp:
            values  = grp["block0_values"][:]          # (T, N) or (N, T)
            axis0   = grp["axis0"][:]                  # column labels
            axis1   = grp["axis1"][:]                  # row index (timestamps)

            # Decode bytes if needed
            if axis0.dtype.kind in ("S", "O"):
                axis0 = [x.decode() if isinstance(x, bytes) else x for x in axis0]
            if axis1.dtype.kind in ("S", "O"):
                axis1 = [x.decode() if isinstance(x, bytes) else x for x in axis1]

            # block0_values is stored as (N, T) in PyTables fixed format — transpose
            if values.shape[0] == len(axis0):
                values = values.T    # now (T, N)

            df = pd.DataFrame(values, index=axis1, columns=axis0)
            print(f"  Reconstructed DataFrame: {df.shape}")

        elif "values_block_0" in grp:
            # Table format
            values = grp["values_block_0"][:]
            axis0  = grp["axis0"][:]
            axis1  = grp["axis1"][:]
            if axis0.dtype.kind in ("S", "O"):
                axis0 = [x.decode() if isinstance(x, bytes) else x for x in axis0]
            if axis1.dtype.kind in ("S", "O"):
                axis1 = [x.decode() if isinstance(x, bytes) else x for x in axis1]
            if values.shape[0] == len(axis0):
                values = values.T
            df = pd.DataFrame(values, index=axis1, columns=axis0)
            print(f"  Reconstructed DataFrame: {df.shape}")

        else:
            # Flat dataset
            ds = f[key]
            if isinstance(ds, h5py.Dataset):
                values = ds[:]
                df = pd.DataFrame(values)
                print(f"  Raw dataset shape: {values.shape}")
            else:
                print("Unrecognised HDF5 layout — cannot auto-fix.")
                return False

    print(f"Writing fixed file to {TMP_PATH} …")
    df.to_hdf(TMP_PATH, key="df", mode="w", format="fixed")

    # Verify the new file is readable
    df_check = pd.read_hdf(TMP_PATH)
    print(f"  Verification OK — shape: {df_check.shape}")

    # Backup original and replace
    os.replace(H5_PATH, BAK_PATH)
    os.replace(TMP_PATH, H5_PATH)
    print(f"Done. Original backed up to {BAK_PATH}")
    print(f"      New file at {H5_PATH}")
    return True


if __name__ == "__main__":
    if not H5_PATH.exists():
        print(f"ERROR: {H5_PATH} not found.")
    else:
        ok = fix_with_h5py()
        if not ok:
            print("\nManual workaround: convert with h5py manually or downgrade pandas.")
