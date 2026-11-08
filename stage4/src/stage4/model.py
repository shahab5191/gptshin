import torch.nn.functional as F
from torch import Tensor, nn


class Mlp(nn.Module):
    block_size: int
    vocab_size: int
    layers: nn.ModuleList

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        embed_dim: int,
        hidden: list[int],
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        dims = [block_size * embed_dim, *hidden, embed_dim]
        linears = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.layers = nn.ModuleList(linears)
        for layer in linears:
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        # embed is tied to the output layer: small init keeps initial logits ~0,
        # otherwise softmax saturates (loss >> ln(vocab)) and the ReLUs die
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        x = x.flatten(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropout(F.relu(x))
        logits = F.linear(x, self.embed.weight)

        return logits

    @property
    def device(self):
        return next(self.parameters()).device
