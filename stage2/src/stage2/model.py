from torch import Tensor, nn


class Mlp(nn.Module):
    dims: list[int]
    layers: nn.ModuleList

    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims
        linears = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.layers = nn.ModuleList(linears)
        for layer in linears:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)

        return x
