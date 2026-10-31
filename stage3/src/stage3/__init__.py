import torch
from torch.cuda import device

from stage3 import spiral
from stage3.model import Mlp
from stage3.train import train


def main() -> None:
    model = Mlp([2, 64, 64, 64, 3])
    spir = spiral.make_spiral()
    x, y = spir
    train(
        model,
        torch.from_numpy(x),
        torch.from_numpy(y),
        steps=1000,
        lr=0.1,
        device="cuda",
    )
    test = torch.Tensor([[0.0, 0.0]]).to("cuda")
    model(test).argmax(dim=1)
    print(x)
