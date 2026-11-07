import torch.nn.functional as F
from torch import Tensor, nn


class Mlp(nn.Module):
    block_size: int
    vocab_size: int
    layers: nn.ModuleList

    def __init__(self, block_size: int, vocab_size: int, hidden: list[int]):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        dims = [block_size * vocab_size, *hidden, vocab_size]
        linears = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.layers = nn.ModuleList(linears)
        for layer in linears:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = F.one_hot(x.long(), self.vocab_size).float().flatten(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)

        return x
