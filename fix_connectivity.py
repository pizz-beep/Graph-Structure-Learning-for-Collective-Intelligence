"""One-shot patch: replace ConnectivityLoss with bounded ReLU version."""
import re

path = r"src\gsl\loss.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# Match the whole ConnectivityLoss class (everything up to the blank line after return)
pattern = re.compile(
    r"class ConnectivityLoss\(nn\.Module\):.*?return penalty\.mean\(\)",
    re.DOTALL,
)

new_class = '''\
class ConnectivityLoss(nn.Module):
    """
    Penalizes isolated nodes — nodes with very low total edge weight.

    Uses a bounded ReLU penalty: loss = mean(relu(target_degree - degree))
    Positive when degree < target_degree, exactly zero otherwise.
    Never goes negative (the old -log formula blew up when top-k produced
    large uniform weights, driving total loss to -600+).

    adj          : (B, N, N)
    target_degree: minimum acceptable mean edge weight per node (default 0.1)
    Output       : scalar >= 0
    """
    def __init__(self, target_degree: float = 0.1):
        super().__init__()
        self.target_degree = target_degree

    def forward(self, adj):
        degree = adj.sum(dim=-1)                        # (B, N)
        # relu: 0 when degree is healthy, positive when too low
        penalty = F.relu(self.target_degree - degree)
        return penalty.mean()'''

result, n = pattern.subn(new_class, src)
if n == 0:
    print("ERROR: pattern not found — printing class block for inspection:")
    start = src.find("class ConnectivityLoss")
    print(repr(src[start:start+600]))
else:
    with open(path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Patched {n} occurrence(s) successfully.")
