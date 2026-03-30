"""Training loop with optional W&B logging."""

import torch
from pathlib import Path
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train(model, train_loader, val_loader, criterion, optimizer, config, device):
    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        totals = {"loss": 0.0, "task_loss": 0.0, "sparsity_loss": 0.0}

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, A = model(x)
            losses = criterion(logits, y, A)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in totals:
                v = losses[k]
                totals[k] += v.item() if hasattr(v, 'item') else v

        for k in totals:
            totals[k] /= len(train_loader)

        val_loss = evaluate(model, val_loader, criterion, device)

        if HAS_WANDB:
            wandb.log({"epoch": epoch, "val/loss": val_loss, **{f"train/{k}": v for k, v in totals.items()}})

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | train={totals['loss']:.4f} | val={val_loss:.4f} | "
                  f"task={totals['task_loss']:.4f} | sparse={totals['sparsity_loss']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path("results").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "results/best_model.pt")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, A = model(x)
            losses = criterion(logits, y, A)
            total += losses["loss"].item()
    return total / len(loader)
