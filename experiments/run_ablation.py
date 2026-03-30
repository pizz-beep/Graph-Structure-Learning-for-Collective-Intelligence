"""
Ablation study: Fixed random graph vs kNN graph vs Learned graph
----------------------------------------------------------------
Run: python experiments/run_ablation.py --config configs/synthetic.yaml

Produces a table + bar chart comparing:
  - Random baseline
  - kNN (k=top_k, using cosine similarity, fixed before training)
  - Learned GSL (end-to-end)

Key metric: graph recovery AUROC + task accuracy
"""

import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gsl.data import SyntheticGraphDataset, get_dataloaders
from src.gsl.model import GSLNet
from src.gsl.loss import JointLoss
from src.gsl.train import train, evaluate
from src.gsl.evaluate import graph_recovery_auroc, task_metrics, plot_adjacency_comparison


def make_fixed_random_adj(N, k, seed=0):
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N))
    for i in range(N):
        nbrs = rng.choice([j for j in range(N) if j != i], size=k, replace=False)
        A[i, nbrs] = 1.0 / k
    return torch.tensor(A, dtype=torch.float32)


def make_knn_adj(features, k):
    """Fixed kNN graph from average node features (cosine similarity)."""
    import torch.nn.functional as F
    x = features.mean(0)                          # [N, F] mean over samples
    x_norm = F.normalize(x, dim=-1)
    S = x_norm @ x_norm.T                         # [N, N]
    S.fill_diagonal_(float('-inf'))
    _, topk_idx = S.topk(k, dim=-1)
    A = torch.zeros_like(S)
    A.scatter_(-1, topk_idx, 1.0 / k)
    return A


class FixedGraphGSLNet(GSLNet):
    """GSLNet with a frozen, pre-specified adjacency (for ablation baselines)."""

    def __init__(self, fixed_A: torch.Tensor, *args, **kwargs):
        # Disable GSL — pass dummy in_features
        kwargs['metric'] = 'cosine'
        super().__init__(*args, **kwargs)
        self.register_buffer('fixed_A', fixed_A)

    def forward(self, x):
        B = x.shape[0] if x.dim() == 3 else None
        A = self.fixed_A
        if B is not None:
            A = A.unsqueeze(0).expand(B, -1, -1)

        import torch.nn.functional as F
        h = F.relu(self.input_proj(x))
        for layer in self.gnn:
            h = layer(h, A)
        if self.task == "graph":
            h = h.mean(dim=-2)
        return self.head(h), A


def run_ablation(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = SyntheticGraphDataset(
        num_nodes=cfg['num_nodes'],
        in_features=cfg['in_features'],
        seed=cfg['seed'],
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, cfg['train_ratio'], cfg['val_ratio'], cfg['batch_size'], cfg['seed']
    )

    results = {}
    criterion = JointLoss(task="classification", sparsity_lambda=cfg['sparsity_lambda'])

    # ---- Variant 1: Random graph ----
    print("\n[1/3] Random graph baseline")
    A_rand = make_fixed_random_adj(cfg['num_nodes'], cfg['top_k'])
    model_rand = FixedGraphGSLNet(
        A_rand, cfg['in_features'], cfg['hidden_dim'], num_classes=5,
        gnn_layers=cfg['gnn_layers'], top_k=cfg['top_k']
    )
    opt = torch.optim.Adam(model_rand.parameters(), lr=cfg['lr'])
    model_rand = train(model_rand, train_loader, val_loader, criterion, opt, cfg, device)
    auroc_rand = graph_recovery_auroc(A_rand, dataset.A_true)
    results['Random'] = {'auroc': auroc_rand}

    # ---- Variant 2: kNN graph (fixed) ----
    print("\n[2/3] Fixed kNN graph")
    A_knn = make_knn_adj(dataset.features, cfg['top_k'])
    model_knn = FixedGraphGSLNet(
        A_knn, cfg['in_features'], cfg['hidden_dim'], num_classes=5,
        gnn_layers=cfg['gnn_layers'], top_k=cfg['top_k']
    )
    opt = torch.optim.Adam(model_knn.parameters(), lr=cfg['lr'])
    model_knn = train(model_knn, train_loader, val_loader, criterion, opt, cfg, device)
    auroc_knn = graph_recovery_auroc(A_knn, dataset.A_true)
    results['kNN (fixed)'] = {'auroc': auroc_knn}

    # ---- Variant 3: Learned GSL ----
    print("\n[3/3] Learned GSL (end-to-end)")
    model_gsl = GSLNet(
        cfg['in_features'], cfg['hidden_dim'], num_classes=5,
        gnn_layers=cfg['gnn_layers'], top_k=cfg['top_k']
    )
    opt = torch.optim.Adam(model_gsl.parameters(), lr=cfg['lr'])
    model_gsl = train(model_gsl, train_loader, val_loader, criterion, opt, cfg, device)

    # Get learned A on a batch of test data
    model_gsl.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(test_loader))
        _, A_learned = model_gsl(x_batch.to(device))
        A_learned = A_learned[0].cpu()  # first in batch

    auroc_gsl = graph_recovery_auroc(A_learned, dataset.A_true)
    results['Learned GSL'] = {'auroc': auroc_gsl}

    # ---- Print results ----
    print("\n" + "="*45)
    print(f"{'Method':<20} {'Recovery AUROC':>14}")
    print("-"*45)
    for name, v in results.items():
        print(f"{name:<20} {v['auroc']:>14.4f}")
    print("="*45)

    # ---- Plot ----
    plot_adjacency_comparison(A_learned, dataset.A_true,
                               title="GSL Recovery: Learned vs Ground Truth",
                               save_path="results/adjacency_comparison.png")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0f0f12')
    ax.set_facecolor('#16161a')
    names = list(results.keys())
    aurocs = [results[n]['auroc'] for n in names]
    colors = ['#3a3a4a', '#1a7a5a', '#5a4ab0']
    bars = ax.bar(names, aurocs, color=colors, edgecolor='#2a2a35')
    ax.axhline(0.5, color='#e07060', linestyle='--', linewidth=1, label='Random chance')
    ax.set_ylabel("Graph Recovery AUROC", color='#9a9aaa')
    ax.set_title("Ablation: Graph Structure Methods", color='#c8c8d8')
    ax.tick_params(colors='#9a9aaa')
    for s in ax.spines.values():
        s.set_edgecolor('#2a2a35')
    ax.legend(facecolor='#16161a', labelcolor='#9a9aaa')
    plt.tight_layout()
    plt.savefig("results/ablation_auroc.png", dpi=150, bbox_inches='tight', facecolor='#0f0f12')
    plt.show()
    print("Saved results/ablation_auroc.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    args = parser.parse_args()
    run_ablation(args.config)
