"""
Evaluation & visualization
--------------------------
graph_recovery_auroc   — the "killer result": AUROC of learned A vs ground truth
task_metrics           — accuracy / F1 on the downstream task
avg_degree             — sparsity of learned graph
plot_adjacency_comparison — heatmap side-by-side
plot_learned_neighborhoods — per-node top-k bar chart (the interpretability slide)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def graph_recovery_auroc(A_pred: torch.Tensor, A_true: torch.Tensor) -> float:
    """AUROC treating edge prediction as binary classification (no supervision on graph)."""
    pred = A_pred.detach().cpu().numpy().flatten()
    true = (A_true.cpu().numpy() > 0).astype(int).flatten()
    N = A_true.shape[0]
    mask = ~np.eye(N, dtype=bool).flatten()
    return roc_auc_score(true[mask], pred[mask])


def regression_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    scaler=None,
) -> dict:
    """
    Compute MAE and RMSE for traffic speed regression.

    Args:
        pred    : (B, out_steps, N) — model predictions (normalized)
        target  : (B, out_steps, N) — ground truth     (normalized)
        scaler  : StandardScaler — if supplied, metrics are in original
                  units (mph). If None, metrics are in normalized units.

    Returns dict with keys 'mae' and 'rmse'.
    """
    p = pred.detach().cpu()
    t = target.detach().cpu()

    if scaler is not None:
        # Convert both back to mph before computing errors
        p = torch.tensor(scaler.inverse_transform(p.numpy()))
        t = torch.tensor(scaler.inverse_transform(t.numpy()))

    # Zero mask — sensors that were offline report 0.0 in the raw data
    mask = (t != 0.0)
    p_valid = p[mask]
    t_valid = t[mask]

    mae  = (p_valid - t_valid).abs().mean().item()
    rmse = ((p_valid - t_valid) ** 2).mean().sqrt().item()

    return {"mae": mae, "rmse": rmse}


# Deprecated alias — kept so old notebooks / scripts don't break immediately
def task_metrics(logits, targets, scaler=None):
    """Deprecated: use regression_metrics() instead."""
    return regression_metrics(logits, targets, scaler)


def avg_degree(A: torch.Tensor, threshold: float = 0.01) -> float:
    return (A > threshold).float().sum(-1).mean().item()


def plot_adjacency_comparison(A_learned, A_true, title="Graph Recovery", save_path=None):
    """Side-by-side heatmap: learned adjacency vs ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0f0f12')

    for ax, (mat, label, cmap) in zip(axes, [
        (A_learned, "Learned A (no graph supervision)", 'magma'),
        (A_true, "Ground Truth (road network)", 'Blues'),
    ]):
        m = mat.detach().cpu().numpy() if hasattr(mat, 'detach') else np.array(mat)
        im = ax.imshow(m, cmap=cmap, aspect='auto')
        ax.set_title(label, color='#c8c8d8', fontsize=11, pad=8)
        ax.set_facecolor('#16161a')
        ax.tick_params(colors='#555568')
        for s in ax.spines.values():
            s.set_edgecolor('#2a2a35')
        plt.colorbar(im, ax=ax)

    fig.suptitle(title, color='#c8c8d8', fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f12')
    plt.show()
    return fig


def plot_learned_neighborhoods(A_learned, node_ids, node_labels=None, top_k=5, save_path=None):
    """
    For each selected node, bar chart of its top-k learned neighbors.
    This is the interpretability result — annotate what the model 'chose' to connect.
    """
    fig, axes = plt.subplots(1, len(node_ids), figsize=(4 * len(node_ids), 4))
    fig.patch.set_facecolor('#0f0f12')
    if len(node_ids) == 1:
        axes = [axes]

    A_np = A_learned.detach().cpu().numpy()

    for ax, node_id in zip(axes, node_ids):
        w = A_np[node_id].copy()
        w[node_id] = 0
        top_nbrs = np.argsort(w)[-top_k:][::-1]
        top_w = w[top_nbrs]

        lbls = [f"Node {n}" if node_labels is None else node_labels[n] for n in top_nbrs[::-1]]
        ax.barh(range(top_k), top_w[::-1], color='#5a4ab0', edgecolor='#a090e0', linewidth=0.5)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(lbls, fontsize=9, color='#9a9aaa')
        ax.set_title(f"Neighbors of Node {node_id}", fontsize=10, color='#c8c8d8')
        ax.set_xlabel("Edge weight", color='#555568')
        ax.set_facecolor('#16161a')
        ax.tick_params(colors='#555568')
        for s in ax.spines.values():
            s.set_edgecolor('#2a2a35')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0f0f12')
    plt.show()
    return fig
