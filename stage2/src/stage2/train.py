import torch
from torch import Tensor, nn, optim

from stage2.model import Mlp


def train(
    model: Mlp, X: Tensor, y: Tensor, steps=1000, lr=0.1, device: str | None = None
) -> tuple[list[float], list[float]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    X = X.to(device)
    y = y.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses: list[float] = []
    accs: list[float] = []
    for i in range(steps):
        optimizer.zero_grad()
        yhat = model(X)
        loss = nn.functional.cross_entropy(yhat, y)
        print(
            f"step {i} loss: {loss.item()} "
            f"accuracy: {yhat.argmax(dim=1).eq(y).float().mean().item()}"
        )
        losses.append(loss.item())
        accs.append(yhat.argmax(dim=1).eq(y).float().mean().item())
        loss.backward()
        optimizer.step()

    return (losses, accs)
