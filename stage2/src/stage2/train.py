import torch
from torch import Tensor, nn, optim

from stage2.model import Mlp


def train(
    model: Mlp,
    X: Tensor,
    y: Tensor,
    steps=1000,
    lr=0.1,
    device: str | None = None,
    batch_size=32,
) -> tuple[list[float], list[float]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    X = X.to(device)
    y = y.to(device)

    n = X.shape[0]
    perm = torch.randperm(n, device=device)
    ptr = 0

    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []
    accs: list[float] = []
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
        losses.append(loss.item())
        accs.append(yhat.argmax(dim=1).eq(yb).float().mean().item())
        loss.backward()
        optimizer.step()

    return (losses, accs)
