import math

import torch
from torch import Tensor, nn, optim

from stage4.model import Mlp
from stage4.utils import KeyListener


@torch.no_grad()
def evaluate(
    model: Mlp, x: Tensor, y: Tensor, batch_size: int = 4096
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].to(model.device).long()
        yb = y[i : i + batch_size].to(model.device).long()
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
) -> tuple[list[float], list[float]]:
    """Train; press "q" in the terminal to stop early.

    With val data and save_path, the checkpoint is written only when val loss
    improves, so save_path always holds the best model seen so far (including
    the starting weights, so resuming never overwrites a better checkpoint).
    """
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
    steps_since_log = 0

    best_val = math.inf
    if val_x is not None:
        best_val, best_ppl = evaluate(model, val_x, val_y)
        print(f"starting val loss {best_val:.4f} (ppl {best_ppl:.1f})")
        if save_path:  # so save_path always exists and holds the best-so-far
            torch.save(model.state_dict(), save_path)

    def validate() -> str:
        nonlocal best_val
        val_loss, val_ppl = evaluate(model, val_x, val_y)
        val_losses.append(val_loss)
        line = f"   val loss {val_loss:.4f} (ppl {val_ppl:.1f})"
        if val_loss < best_val:
            best_val = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                line += f"  * best, saved to {save_path}"
        return line

    print('training - press "q" to stop early')
    with KeyListener() as keys:
        for i in range(steps):
            if ptr + batch_size > n:
                perm = torch.randperm(n)
                ptr = 0
            idx = perm[ptr : ptr + batch_size]
            ptr += batch_size
            xb = X[idx].to(device).long()
            yb = y[idx].to(device).long()
            optimizer.zero_grad()
            yhat = model(xb)
            loss = nn.functional.cross_entropy(yhat, yb)
            loss_v = loss.item()
            losses.append(loss_v)
            running_loss += loss_v
            steps_since_log += 1
            loss.backward()
            optimizer.step()

            stop = keys.pressed("q")
            if (i + 1) % log_every == 0 or stop:
                mean_loss = running_loss / steps_since_log
                line = (
                    f"step {i + 1:>7}/{steps}: "
                    f"train loss {mean_loss:.4f} (ppl {math.exp(mean_loss):.1f})"
                )
                if val_x is not None:
                    line += validate()
                print(line)
                running_loss = 0.0
                steps_since_log = 0
            if stop:
                print(f"stopped by user at step {i + 1}")
                break

    if val_x is not None:
        print(f"best val loss {best_val:.4f} (ppl {math.exp(best_val):.1f})")
    return (losses, val_losses)
