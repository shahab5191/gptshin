import torch

from stage2.model import Mlp
from stage2.spiral import make_spiral
from stage2.train import train


def main() -> None:
    spiral = make_spiral()
    model = Mlp([2, 64, 64, 32, 16, 8, 3])
    # make tensor from spiral[0]
    x = torch.from_numpy(spiral[0]).float()
    y = torch.from_numpy(spiral[1])

    train(model, x, y, 1000, 0.1)
