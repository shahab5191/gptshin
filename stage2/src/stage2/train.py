from torch import Tensor, nn, optim

from stage2.model import Mlp


def train(model: Mlp, X: Tensor, y: Tensor, steps=1000, lr=0.1) -> None:
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for i in range(steps):
        optimizer.zero_grad()
        yhat = model(X)
        loss = nn.functional.cross_entropy(yhat, y)
        print(f"step {i} loss: {loss}")
        loss.backward()
        optimizer.step()
