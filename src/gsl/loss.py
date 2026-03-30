"""
Joint Loss:  L = L_task + λ * L_sparsity

L_task:     Cross-entropy (classification) or MSE (regression)
L_sparsity: L1 norm on adjacency to encourage sparse communication graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    def __init__(
        self,
        task: str = "classification",
        sparsity_lambda: float = 0.001,
        sparsity_norm: str = "l1",    # "l1" | "nuclear"
    ):
        super().__init__()
        self.task = task
        self.lam = sparsity_lambda
        self.sparsity_norm = sparsity_norm

    def forward(self, logits, targets, A) -> dict:
        if self.task == "classification":
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1).long()
            )
        else:
            task_loss = F.mse_loss(logits.squeeze(-1), targets.float())

        if self.sparsity_norm == "l1":
            sparsity_loss = A.abs().mean()
        else:  # nuclear norm
            if A.dim() == 2:
                sparsity_loss = torch.linalg.matrix_norm(A, ord='nuc') / A.size(0)
            else:
                sparsity_loss = torch.stack([
                    torch.linalg.matrix_norm(A[b], ord='nuc')
                    for b in range(A.size(0))
                ]).mean() / A.size(1)

        total = task_loss + self.lam * sparsity_loss
        return {
            "loss": total,
            "task_loss": task_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
        }
