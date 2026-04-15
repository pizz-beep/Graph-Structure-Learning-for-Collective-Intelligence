# src/gsl/train.py

"""Training loop with optional W&B logging."""

import torch
from pathlib import Path
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ─────────────────────────────────────────────
# 1. Evaluation loop
# ─────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    """
    Runs one full pass over loader in eval mode.
    Returns average total loss (scalar float).

    We disable grad here because we're only reading loss values —
    no backward pass, no parameter updates.
    """
    model.eval()
    total = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # model returns three values now
            pred, adj, node_emb = model(x)

            # criterion returns (scalar, components_dict)
            loss, _ = criterion(pred, y, adj, node_emb)
            total += loss.item()

    return total / len(loader)


# ─────────────────────────────────────────────
# 2. Training loop
# ─────────────────────────────────────────────

def train(model, train_loader, val_loader, criterion, optimizer, config, device):
    """
    Full training run.

    config keys expected:
        epochs        : int
        save_dir      : str  — where to write best_model.pt
        log_every     : int  — print interval in epochs (default 10)

    Returns the trained model (with best weights loaded).
    """
    model = model.to(device)
    best_val_loss = float('inf')
    save_dir = Path(config.get('save_dir', 'results'))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_every = config.get('log_every', 10)

    for epoch in range(config['epochs']):
        model.train()

        # Accumulate every loss component separately so you can
        # watch them individually in W&B and diagnose which term
        # is dominating
        totals = {
            "loss/total":        0.0,
            "loss/task":         0.0,
            "loss/sparsity":     0.0,
            "loss/smoothness":   0.0,
            "loss/connectivity": 0.0,
        }

        for x, y in tqdm(train_loader,
                         desc=f"Epoch {epoch+1}/{config['epochs']}",
                         leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            # Unpack all three return values from model.forward
            pred, adj, node_emb = model(x)

            # criterion returns (total_loss_tensor, components_dict)
            loss, components = criterion(pred, y, adj, node_emb)

            loss.backward()

            # Gradient clipping prevents exploding gradients —
            # especially important with the GCN since adjacency
            # weights can amplify gradients across layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Accumulate all five components
            for k in totals:
                totals[k] += components[k]

        # Average over batches
        n = len(train_loader)
        for k in totals:
            totals[k] /= n

        val_loss = evaluate(model, val_loader, criterion, device)

        # ── Logging ──────────────────────────────────────
        if HAS_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "val/loss": val_loss,
                **{f"train/{k}": v for k, v in totals.items()}
            })

        if (epoch + 1) % log_every == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"train={totals['loss/total']:.4f} | "
                f"val={val_loss:.4f} | "
                f"task={totals['loss/task']:.4f} | "
                f"sparse={totals['loss/sparsity']:.4f} | "
                f"smooth={totals['loss/smoothness']:.4f}"
            )

        # ── Checkpoint ───────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / "best_model.pt"
            torch.save({
                "epoch":      epoch + 1,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   best_val_loss,
                "config":     config,
            }, ckpt_path)
            print(f"  [saved] checkpoint (val={best_val_loss:.4f})")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")

    # Load best weights before returning so the caller always
    # gets the best model, not the last epoch's model
    best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    return model