"""Patch GSLLoss.__init__ so it instantiates ConnectivityLoss() with no args (uses default target_degree=0.1)."""
import re

path = r"src\gsl\loss.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# The old line was:  self.connectivity = ConnectivityLoss()
# It already calls ConnectivityLoss() with no args, which now uses __init__(target_degree=0.1)
# So actually no change needed — just verify it's there
if "self.connectivity = ConnectivityLoss()" in src:
    print("ConnectivityLoss() instantiation is already correct (uses default target_degree=0.1).")
else:
    print("WARNING: could not find ConnectivityLoss() instantiation line.")

# Also verify the new class is in place
if "F.relu(self.target_degree - degree)" in src:
    print("New bounded ReLU formula is present. Patch confirmed.")
else:
    print("ERROR: new formula NOT found.")

# Quick sanity: import the module
import sys
sys.path.insert(0, "src")
# Force reload
import importlib
import gsl.loss
importlib.reload(gsl.loss)
from gsl.loss import ConnectivityLoss, GSLLoss
import torch

adj = torch.rand(4, 10, 10)
c = ConnectivityLoss()
val = c(adj)
print(f"ConnectivityLoss output: {val.item():.6f}  (must be >= 0)")
assert val.item() >= 0, "STILL NEGATIVE!"

crit = GSLLoss(task="classification")
print(f"GSLLoss instantiated OK.")
print("ALL CHECKS PASSED")
