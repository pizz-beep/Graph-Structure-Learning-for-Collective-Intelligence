"""
Graph Structure Learner
-----------------------
Given node features X ∈ R^{N x F}, learns a sparse adjacency matrix A ∈ R^{N x N}.

Two similarity metric modes:
  - cosine: fast, parameter-free
  - mlp:    learned pairwise metric (more expressive)

Sparsification modes:
  - top_k:  keep top-k neighbours per node (hard threshold)
  - gumbel: Gumbel-softmax relaxation (differentiable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphStructureLearner(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        metric: str = "cosine",       # "cosine" | "mlp"
        sparsify: str = "top_k",      # "top_k" | "gumbel"
        top_k: int = 10,
        gumbel_tau: float = 0.5,
    ):
        super().__init__()
        self.metric = metric
        self.sparsify = sparsify
        self.top_k = top_k
        self.tau = gumbel_tau

        if metric == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.scorer = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, F] or [B, N, F]
        Returns:
            A: Soft adjacency matrix [N, N] or [B, N, N]
        """
        batched = x.dim() == 3
        if not batched:
            x = x.unsqueeze(0)

        B, N, F = x.shape

        # Step 1: pairwise similarity scores
        if self.metric == "cosine":
            x_norm = F.normalize(x, dim=-1)
            S = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

        elif self.metric == "mlp":
            h = self.encoder(x)                              # [B, N, H]
            hi = h.unsqueeze(2).expand(-1, -1, N, -1)
            hj = h.unsqueeze(1).expand(-1, N, -1, -1)
            pairs = torch.cat([hi, hj], dim=-1)
            S = self.scorer(pairs).squeeze(-1)               # [B, N, N]
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Remove self-loops
        eye = torch.eye(N, device=x.device).bool().unsqueeze(0)
        S = S.masked_fill(eye, float('-inf'))

        # Step 2: sparsification
        if self.sparsify == "top_k":
            A = self._top_k_sparsify(S, self.top_k)
        elif self.sparsify == "gumbel":
            A = self._gumbel_sparsify(S, self.tau)
        else:
            raise ValueError(f"Unknown sparsify: {self.sparsify}")

        if not batched:
            A = A.squeeze(0)
        return A

    @staticmethod
    def _top_k_sparsify(S: torch.Tensor, k: int) -> torch.Tensor:
        """Keep top-k entries per row; softmax over selected entries."""
        B, N, _ = S.shape
        _, topk_idx = S.topk(k, dim=-1)

        mask = torch.zeros_like(S)
        mask.scatter_(-1, topk_idx, 1.0)

        S_masked = S * mask + (1 - mask) * float('-inf')
        A = torch.softmax(S_masked, dim=-1)
        return A * mask

    @staticmethod
    def _gumbel_sparsify(S: torch.Tensor, tau: float) -> torch.Tensor:
        """Gumbel-softmax: differentiable discrete approximation."""
        if S.requires_grad:
            gumbel = -torch.log(-torch.log(torch.rand_like(S) + 1e-10) + 1e-10)
            S = S + gumbel
        return torch.softmax(S / tau, dim=-1)
