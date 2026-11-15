import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Mlp(nn.Module):
    block_size: int
    vocab_size: int

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        embed_dim: int,
        hidden_w: int,
        hidden_c: int,
        residual: bool = True,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.residual = residual
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.in_proj = nn.Linear(embed_dim, hidden_w)
        self.out_proj = nn.Linear(hidden_w, embed_dim)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
            ),
        )

        nn.init.kaiming_normal_(self.W_q.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.W_q.bias)
        nn.init.kaiming_normal_(self.in_proj.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.in_proj.bias)
        nn.init.kaiming_normal_(
            self.out_proj.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.zeros_(self.out_proj.bias)

        if residual:
            blocks = []
            for _ in range(hidden_c):
                inner = nn.Linear(hidden_w, hidden_w)
                nn.init.kaiming_normal_(
                    inner.weight, mode="fan_in", nonlinearity="relu"
                )
                nn.init.zeros_(inner.bias)

                outer = nn.Linear(hidden_w, hidden_w)
                nn.init.zeros_(outer.weight)
                nn.init.zeros_(outer.bias)

                blocks.append(nn.Sequential(inner, nn.ReLU(), outer))
            self.blocks = nn.ModuleList(blocks)
        else:
            layers = [nn.Linear(hidden_w, hidden_w) for _ in range(hidden_c)]
            for layer in layers:
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )
                nn.init.zeros_(layer.bias)
            self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x) + self.pos_embed.weight
        q = self.W_q(x)
        w = torch.einsum("btd,bsd->bts", q, x)
        w = w / (self.embed_dim**0.5)
        w = w.masked_fill(self.causal_mask, float("-inf"))
        w = w.softmax(dim=-1)
        x = torch.einsum("bts,bsd->btd", w, x)
        x = F.relu(self.in_proj(x))
        if self.residual:
            for block in self.blocks:
                x = x + block(x)
        else:
            for layer in self.layers:
                x = F.relu(layer(x))
        x = self.out_proj(x)
        logits = F.linear(x, self.embed.weight)
        return logits

    @property
    def device(self):
        return next(self.parameters()).device
