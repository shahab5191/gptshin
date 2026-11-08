import math

import torch
from torch import Tensor, nn, optim

from stage4.model import Mlp


@torch.no_grad()
def evaluate(
    model: Mlp, x: Tensor, y: Tensor, batch_size: int = 4096
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].to(model.device)
        yb = y[i : i + batch_size].to(model.device)
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb, reduction="sum")
        total_loss += loss.item()
        total += yb.shape[0]
    model.train()
    mean_loss = total_loss / total
    return mean_loss, math.exp(mean_loss)


def train(
    model: Mlp,
    X: Tensor,
    y: Tensor,
    val_x: Tensor | None = None,
    val_y: Tensor | None = None,
    steps=1000,
    lr=0.1,
    device: str | None = None,
    batch_size=512,
    log_every=1000,
    weight_decay=0.01,
    save_path: str | None = None,
    save_every: int = 0,
) -> tuple[list[float], list[float]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n = X.shape[0]
    perm = torch.randperm(n)
    ptr = 0

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    val_losses: list[float] = []
    running_loss = 0.0
    for i in range(steps):
        if ptr + batch_size > n:
            perm = torch.randperm(n)
            ptr = 0
        idx = perm[ptr : ptr + batch_size]
        ptr += batch_size
        xb = X[idx].to(device)
        yb = y[idx].to(device)
        optimizer.zero_grad()
        yhat = model(xb)
        loss = nn.functional.cross_entropy(yhat, yb)
        loss_v = loss.item()
        losses.append(loss_v)
        running_loss += loss_v
        loss.backward()
        optimizer.step()
        if save_path and save_every and (i + 1) % save_every == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model Stored at {save_path}")
        if (i + 1) % log_every == 0:
            mean_loss = running_loss / log_every
            line = (
                f"step {i + 1:>7}/{steps}: "
                f"train loss {mean_loss:.4f} (ppl {math.exp(mean_loss):.1f})"
            )
            if val_x is not None:
                val_loss, val_ppl = evaluate(model, val_x, val_y)
                val_losses.append(val_loss)
                line += f"   val loss {val_loss:.4f} (ppl {val_ppl:.1f})"
            print(line)
            running_loss = 0.0

    return (losses, val_losses)
