# src/gsl/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Prediction loss
# ─────────────────────────────────────────────

class MaskedMAELoss(nn.Module):
    """
    Mean Absolute Error that ignores zero-valued entries.

    Why mask zeros? METR-LA contains zeros where sensors were offline
    or malfunctioning. Including those in the loss would punish the model
    for something that isn't real signal — it would learn to predict zero
    in ambiguous situations rather than learn the traffic pattern.

    Input:
        pred  : (B, out_steps, N)  — model predictions (normalized)
        target: (B, out_steps, N)  — ground truth (normalized)
    Output:
        scalar loss value
    """
    def __init__(self, null_val=0.0):
        super().__init__()
        self.null_val = null_val

    def forward(self, pred, target):
        # Build mask: True where target is NOT the null value
        mask = (target != self.null_val).float()

        # MAE only over valid entries
        loss = torch.abs(pred - target) * mask

        # Normalize by number of valid entries (not total entries)
        return loss.sum() / mask.sum().clamp(min=1)


class MaskedRMSELoss(nn.Module):
    """
    Root Mean Squared Error with the same zero-masking logic.
    RMSE penalizes large errors more heavily than MAE.
    We use MAE as the training loss and RMSE as an evaluation metric —
    training on RMSE can cause the model to over-focus on rare spikes.
    """
    def __init__(self, null_val=0.0):
        super().__init__()
        self.null_val = null_val

    def forward(self, pred, target):
        mask = (target != self.null_val).float()
        loss = ((pred - target) ** 2) * mask
        return torch.sqrt(loss.sum() / mask.sum().clamp(min=1))


# ─────────────────────────────────────────────
# 2. Graph regularization losses
# ─────────────────────────────────────────────

class SparsityLoss(nn.Module):
    """
    L1 penalty on the adjacency matrix entries.

    Pushes edge weights toward zero — the model has to "pay" to use
    an edge, so it only keeps connections that genuinely help prediction.
    Without this, the model tends to keep a dense graph because that's
    the path of least resistance.

    adj   : (B, N, N)
    Output: scalar — mean absolute value of all edge weights
    """
    def forward(self, adj):
        return adj.abs().mean()


class SmoothnessLoss(nn.Module):
    """
    Graph Laplacian smoothness regularizer.

    Encourages connected nodes to have similar feature values.
    Formula: (1/2) * sum_{i,j} A_{ij} * ||h_i - h_j||^2
           = tr(H^T L H)  where L = D - A is the graph Laplacian

    In plain English: if two nodes are strongly connected in your
    learned graph, their feature representations should be similar.
    This prevents the model from creating high-weight edges between
    nodes whose signals don't actually resemble each other.

    adj     : (B, N, N)  — learned adjacency
    node_emb: (B, N, D)  — node embeddings from the encoder
    Output  : scalar
    """
    def forward(self, adj, node_emb):
        B, N, D = node_emb.shape

        # Direct pairwise smoothness: sum_{i,j} A_{ij} * ||h_i - h_j||^2
        #
        # The Laplacian approach (tr(H^T L H)) requires A to be SYMMETRIC and
        # PSD. TopKSparsifier breaks symmetry (node i's top-k != node j's top-k),
        # so L = D - A is no longer PSD and the quadratic form goes to -inf.
        #
        # This formula is ALWAYS >= 0:  A_{ij} >= 0 (after ReLU) and
        # squared distances >= 0 by definition.  No symmetry needed.
        #
        # Equivalent identity: ||h_i - h_j||^2 = ||h_i||^2 + ||h_j||^2 - 2*h_i.h_j
        h_sq = (node_emb ** 2).sum(-1, keepdim=True)            # (B, N, 1)
        sq_dist = (h_sq + h_sq.transpose(1, 2)
                   - 2 * torch.bmm(node_emb, node_emb.transpose(1, 2)))  # (B, N, N)
        sq_dist = sq_dist.clamp(min=0)   # guard against tiny negative floats

        # Normalize by B*N so the magnitude doesn't scale with batch/graph size
        return (adj * sq_dist).sum() / (B * N)



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
        return penalty.mean()


# ─────────────────────────────────────────────
# 3. Joint loss
# ─────────────────────────────────────────────

class GSLLoss(nn.Module):
    """
    The combined loss that trains the whole system end-to-end.

    Total loss = task_loss
               + lambda_s  * sparsity_loss
               + lambda_sm * smoothness_loss
               + lambda_c  * connectivity_loss

    Args:
        task       : "regression" (default, uses MaskedMAE — for METR-LA)
                     "classification" (uses CrossEntropy — for synthetic / Cora)
        lambda_s   : sparsity penalty weight
        lambda_sm  : smoothness penalty weight
        lambda_c   : connectivity penalty weight

    For regression:
        pred   : (B, out_steps, N)  — continuous predictions
        target : (B, out_steps, N)  — continuous targets

    For classification:
        pred   : (B, N, C)          — class logits
        target : (B, N)             — integer class indices (long)
    """
    def __init__(self, task="regression",
                 lambda_s=0.001, lambda_sm=0.01, lambda_c=0.001):
        super().__init__()
        self.task      = task
        self.lambda_s  = lambda_s
        self.lambda_sm = lambda_sm
        self.lambda_c  = lambda_c

        if task == "regression":
            self.task_loss = MaskedMAELoss()
        elif task == "classification":
            self.task_loss = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"task must be 'regression' or 'classification', got '{task}'")

        self.sparsity     = SparsityLoss()
        self.smoothness   = SmoothnessLoss()
        self.connectivity = ConnectivityLoss()

    def forward(self, pred, target, adj, node_emb):
        if self.task == "classification":
            # CrossEntropyLoss expects (B*N, C) and (B*N,)
            B, N, C = pred.shape
            l_task = self.task_loss(
                pred.reshape(B * N, C),
                target.reshape(B * N).long(),
            )
        else:
            l_task = self.task_loss(pred, target)

        l_sparse = self.sparsity(adj)
        l_smooth = self.smoothness(adj, node_emb)
        l_conn   = self.connectivity(adj)

        total = (l_task
                 + self.lambda_s  * l_sparse
                 + self.lambda_sm * l_smooth
                 + self.lambda_c  * l_conn)

        components = {
            "loss/total":        total.item(),
            "loss/task":         l_task.item(),
            "loss/sparsity":     l_sparse.item(),
            "loss/smoothness":   l_smooth.item(),
            "loss/connectivity": l_conn.item(),
        }

        return total, components


# Alias so existing imports (`from .loss import JointLoss`) continue to work
JointLoss = GSLLoss