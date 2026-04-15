# src/gsl/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. Similarity metric
# ─────────────────────────────────────────────

class CosineSimilarityMetric(nn.Module):
    """
    Computes pairwise cosine similarity between all N node embeddings.

    Input  : node_emb  (B, N, D)  — a batch of N nodes, each with D features
    Output : sim_matrix (B, N, N) — sim_matrix[b, i, j] is how similar
                                     node i and node j are in sample b

    Cosine similarity is a natural first choice because it's scale-invariant —
    two sensors with the same *pattern* of readings but different absolute
    speeds will still score high similarity. That's what we want: structural
    similarity, not magnitude matching.
    """
    def forward(self, node_emb):
        # L2-normalize each node embedding along the feature dimension
        norm = F.normalize(node_emb, p=2, dim=-1)   # (B, N, D)
        # Batch matrix multiply: (B, N, D) x (B, D, N) → (B, N, N)
        sim = torch.bmm(norm, norm.transpose(1, 2))
        return sim                                   # values in [-1, 1]


class MLPSimilarityMetric(nn.Module):
    """
    Learns the similarity metric itself rather than hard-coding cosine.

    Takes two node embeddings [h_i, h_j] concatenated → scalar score.
    This lets the model discover asymmetric relationships:
    sensor A being near sensor B doesn't have to mean B is near A
    in the task-relevant sense (e.g. upstream vs downstream traffic).

    Input  : node_emb  (B, N, D)
    Output : sim_matrix (B, N, N)

    Note: this is O(N²) MLP calls — fine for N=207, expensive for N=10000.
    """
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_emb):
        B, N, D = node_emb.shape

        # Expand to get all (i, j) pairs in one shot
        h_i = node_emb.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
        h_j = node_emb.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)

        pair = torch.cat([h_i, h_j], dim=-1)             # (B, N, N, 2D)
        scores = self.mlp(pair).squeeze(-1)               # (B, N, N)
        return scores


# ─────────────────────────────────────────────
# 2. Sparsification
# ─────────────────────────────────────────────

class TopKSparsifier(nn.Module):
    """
    Hard sparsification: for each node i, keep only its top-k most
    similar neighbors, zero out the rest.

    Why sparsify? Dense graphs mean every sensor talks to every other sensor.
    That's unrealistic and kills interpretability — you can't look at a
    207×207 fully-connected matrix and learn anything. A sparse graph
    forces the model to commit to meaningful connections.

    Problem: top-k is not differentiable (argmax has zero gradient).
    Solution: we keep the top-k *mask* but multiply it against the
    original soft scores, so gradients still flow through the scores
    that survive. This is the straight-through estimator trick.

    Input  : sim_matrix (B, N, N)
    Output : sparse_adj (B, N, N) — same values in top-k positions, 0 elsewhere
    """
    def __init__(self, k=10):
        super().__init__()
        self.k = k

    def forward(self, sim_matrix):
        B, N, _ = sim_matrix.shape
        k = min(self.k, N)

        # Find indices of top-k values per row
        topk_vals, topk_idx = torch.topk(sim_matrix, k, dim=-1)  # (B, N, k)

        # Build a binary mask of the same shape as sim_matrix
        mask = torch.zeros_like(sim_matrix)
        mask.scatter_(-1, topk_idx, 1.0)                          # (B, N, N)

        # Apply mask — gradients flow through sim_matrix, not through mask
        sparse_adj = sim_matrix * mask
        return sparse_adj


class GumbelSoftmaxSparsifier(nn.Module):
    """
    Soft, fully differentiable sparsification using Gumbel-Softmax.

    Instead of a hard 0/1 mask, this produces soft edge weights that
    approach discrete during inference (low temperature) but are smooth
    during training (higher temperature), giving clean gradients.

    Why this over top-k? Top-k has a discontinuity — a small change in
    a score can suddenly include or exclude a neighbor, causing gradient
    instability. Gumbel-Softmax avoids this. Use it when training is
    unstable with top-k.

    temperature: start high (~1.0) and anneal toward 0.1 during training.
    Hard mode (hard=True): straight-through in forward, soft in backward.

    Input  : sim_matrix (B, N, N)
    Output : soft_adj   (B, N, N)
    """
    def __init__(self, temperature=0.5, hard=False):
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, sim_matrix):
        return F.gumbel_softmax(sim_matrix, tau=self.temperature,
                                hard=self.hard, dim=-1)


# ─────────────────────────────────────────────
# 3. Graph convolution layer
# ─────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """
    One layer of spectral-style graph convolution.

    The classic GCN update rule (Kipf & Welling 2017):
        H' = σ( D^{-1/2} A D^{-1/2} H W )

    In plain English:
        1. Normalize the adjacency matrix by node degree
           (so nodes with many neighbors don't dominate)
        2. Aggregate: each node collects a weighted average of its neighbors
        3. Transform: apply a learnable linear projection W
        4. Activate: pass through ReLU

    Here A is your *learned* adjacency from the GSL module, not a fixed graph.
    This is the key coupling: the structure learner and the GNN are
    trained jointly, so the graph shape adapts to minimize task loss.

    Input:
        x   : (B, N, in_features)
        adj : (B, N, N)  — learned soft adjacency (not necessarily symmetric)
    Output:
        out : (B, N, out_features)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, adj):
        # Step 1: degree normalization
        # Sum each row of adj to get degree vector, then invert
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)   # (B, N, 1)
        adj_norm = adj / deg                                    # (B, N, N)

        # Step 2: aggregate neighbor features
        # (B, N, N) x (B, N, F_in) → (B, N, F_in)
        agg = torch.bmm(adj_norm, x)

        # Step 3: linear transform + activation
        out = F.relu(self.W(agg))                              # (B, N, F_out)
        return out


# ─────────────────────────────────────────────
# 4. Node feature encoder
# ─────────────────────────────────────────────

class NodeEncoder(nn.Module):
    """
    Projects raw node features into a richer embedding space before
    computing pairwise similarity.

    Why not compute similarity directly on raw features?
    Raw speed readings are 1D scalars. The similarity metric works better
    in a higher-dimensional space where the model can learn to separate
    "structurally similar" from "currently similar" — two sensors that
    always co-vary are more meaningful neighbors than two that just happen
    to have the same speed right now.

    Input  : x   (B, N, in_steps)  — the raw time window per node
    Output : emb (B, N, hidden_dim)
    """
    def __init__(self, in_steps, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)   # (B, N, hidden_dim)

class GraphStructureLearner(nn.Module):
    """
    Wires together: NodeEncoder → SimilarityMetric → Sparsifier
    
    This is the full GSL pipeline in one module.
    Input : x   (B, N, in_steps)  — raw node features
    Output: adj (B, N, N)         — learned sparse adjacency
    """
    def __init__(self, in_features, hidden_dim=64,
                 metric="cosine", sparsify="top_k", top_k=10,
                 temperature=0.5):
        super().__init__()

        self.encoder = NodeEncoder(in_features, hidden_dim)

        if metric == "cosine":
            self.metric = CosineSimilarityMetric()
        elif metric == "mlp":
            self.metric = MLPSimilarityMetric(hidden_dim)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if sparsify == "top_k":
            self.sparsifier = TopKSparsifier(k=top_k)
        elif sparsify == "gumbel":
            self.sparsifier = GumbelSoftmaxSparsifier(temperature=temperature)
        else:
            raise ValueError(f"Unknown sparsifier: {sparsify}")

    def forward(self, x):
        emb = self.encoder(x)           # (B, N, hidden_dim)
        sim = self.metric(emb)          # (B, N, N)
        adj = self.sparsifier(sim)      # (B, N, N) sparse
        return adj, emb                 # return emb too — loss.py needs it