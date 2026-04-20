"""
GSLNet: Graph Structure Learning Network
-----------------------------------------
Pipeline: Node features → GraphStructureLearner → GNN → Task head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphStructureLearner, GraphConvLayer

class GSLNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_classes: int,
        gnn_layers: int = 2,
        metric: str = "cosine",
        sparsify: str = "top_k",
        top_k: int = 10,
        task: str = "node",           # "node" | "graph"
    ):
        super().__init__()

        self.gsl = GraphStructureLearner(
            in_features=in_features,
            hidden_dim=hidden_dim,
            metric=metric,
            sparsify=sparsify,
            top_k=top_k,
        )

        self.gnn = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim)
            for _ in range(gnn_layers)
        ])

        self.task = task
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        x comes from SlidingWindowDataset as (B, in_steps, N).
        NodeEncoder / GraphStructureLearner expect (B, N, in_steps),
        so we transpose here once and keep the rest of the pipeline unchanged.

        Returns:
            logits: [B, N, C] for node task  (or [B, C] for graph task)
            A:      learned adjacency (B, N, N) — used in loss + visualization
            node_emb: (B, N, hidden_dim) — returned for loss regularisation
        """
        # (B, in_steps, N) → (B, N, in_steps)
        x = x.transpose(1, 2)

        A, node_emb = self.gsl(x)
        h = node_emb

        for layer in self.gnn:
            h = layer(h, A)

        if self.task == "graph":
            h = h.mean(dim=-2)

        logits = self.head(h)  # (B, N, out_steps)

        # Regression loss expects (B, out_steps, N) to align with DataLoader y
        if self.task != "graph":
            logits = logits.transpose(1, 2)   # (B, out_steps, N)

        return logits, A, node_emb

    @torch.no_grad()
    def get_learned_graph(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the adjacency matrix (discards node_emb)."""
        adj, _ = self.gsl(x)
        return adj
