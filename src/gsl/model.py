"""
GSLNet: Graph Structure Learning Network
-----------------------------------------
Pipeline: Node features → GraphStructureLearner → GNN → Task head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphStructureLearner


class GNNLayer(nn.Module):
    """Single message-passing layer using a learned adjacency A."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Aggregate: h_i = sum_j A_ij * x_j
        if x.dim() == 2:
            agg = torch.mm(A, x)
        else:
            agg = torch.bmm(A, x)
        return F.relu(self.norm(self.lin(agg)))


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

        self.input_proj = nn.Linear(in_features, hidden_dim)

        self.gnn = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim)
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
        Returns:
            logits: [N, C] or [B, N, C] for node task
            A:      learned adjacency (used in loss + visualization)
        """
        A = self.gsl(x)
        h = F.relu(self.input_proj(x))

        for layer in self.gnn:
            h = layer(h, A)

        if self.task == "graph":
            h = h.mean(dim=-2)

        return self.head(h), A

    @torch.no_grad()
    def get_learned_graph(self, x: torch.Tensor) -> torch.Tensor:
        return self.gsl(x)
