import torch
from torch import Tensor, nn, optim

from stage3.model import Mlp


def train(
    model: Mlp,
    X: Tensor,
    y: Tensor,
    steps=1000,
    lr=0.1,
    device: str | None = None,
    batch_size=512,
    log_every=1000,
    weight_decay=0.01,
) -> tuple[list[float], list[float]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    X = X.to(device)
    y = y.to(device)

    n = X.shape[0]
    perm = torch.randperm(n, device=device)
    ptr = 0

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    accs: list[float] = []
    running_loss = 0.0
    running_acc = 0.0
    for i in range(steps):
        if ptr + batch_size > n:
            perm = torch.randperm(n, device=device)
            ptr = 0
        idx = perm[ptr : ptr + batch_size]
        ptr += batch_size
        xb, yb = X[idx], y[idx]
        optimizer.zero_grad()
        yhat = model(xb)
        loss = nn.functional.cross_entropy(yhat, yb)
        loss_v = loss.item()
        acc_v = yhat.argmax(dim=1).eq(yb).float().mean().item()
        losses.append(loss_v)
        accs.append(acc_v)
        running_loss += loss_v
        running_acc += acc_v
        loss.backward()
        optimizer.step()
        if (i + 1) % log_every == 0:
            print(
                f"step {i + 1:>7}/{steps}: "
                f"loss {running_loss / log_every:.4f}  "
                f"acc {running_acc / log_every:.4f}"
            )
            running_loss = 0.0
            running_acc = 0.0

    return (losses, accs)
