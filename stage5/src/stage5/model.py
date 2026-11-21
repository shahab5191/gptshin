import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    """Multi-head causal self-attention with QKV projections and an output proj.

    [B, T, D] -> [B, T, D]: each position becomes a weighted average of the
    positions it may see, with weights computed on the fly from q·k. `W_o` starts
    at zero so a fresh attention block adds nothing to the stream.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads:
            raise Exception("Embed dimension should be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        for proj in (self.W_q, self.W_k, self.W_v):
            nn.init.kaiming_normal_(proj.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(proj.bias)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        B, T, D = x.shape
        H = self.num_heads
        dh = self.head_dim

        q = self.W_q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, dh).transpose(1, 2)

        scores = torch.einsum("bhtd,bhsd->bhts", q, k) / (dh**0.5)
        scores = scores.masked_fill(causal_mask[:T, :T], float("-inf"))
        attn = scores.softmax(dim=-1)

        out = torch.einsum("bhts,bhsd->bhtd", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class MLP(nn.Module):
    """Per-position MLP: D -> H, `n_blocks` residual blocks at width H, H -> D.

    out_proj starts at zero, so a fresh MLP adds nothing to the stream and the
    whole stack starts close to an identity.
    """

    def __init__(self, embed_dim: int, hidden_w: int, n_blocks: int):
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, hidden_w)
        self.out_proj = nn.Linear(hidden_w, embed_dim)
        nn.init.kaiming_normal_(self.in_proj.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        blocks = []
        for _ in range(n_blocks):
            inner = nn.Linear(hidden_w, hidden_w)
            nn.init.kaiming_normal_(inner.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(inner.bias)

            outer = nn.Linear(hidden_w, hidden_w)
            nn.init.zeros_(outer.weight)
            nn.init.zeros_(outer.bias)

            blocks.append(nn.Sequential(inner, nn.ReLU(), outer))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.in_proj(x))
        for block in self.blocks:
            x = x + block(x)
        return self.out_proj(x)


class Layer(nn.Module):
    """Mix across positions, then process each position - both as residuals.

    Pre-norm: each branch reads a LayerNorm'd copy of the stream, so its input
    has a fixed scale no matter how large the stream has grown. The stream
    itself stays a plain sum (x + ...), keeping a direct gradient path.
    """

    def __init__(self, embed_dim: int, hidden_w: int, n_blocks: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_w, n_blocks)

    def forward(self, x: Tensor, causal_mask: Tensor) -> Tensor:
        # attn uses its input as query source, keys AND values - all normalized
        x = x + self.attn(self.norm1(x), causal_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Model(nn.Module):
    """embed + pos -> [LN -> Attention, LN -> MLP] x n_layers -> LN -> tied output."""

    block_size: int
    vocab_size: int

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        embed_dim: int,
        hidden_w: int,
        blocks_per_layer: int,
        n_layers: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        nn.init.normal_(self.embed.weight, std=embed_dim**-0.5)
        nn.init.normal_(self.pos_embed.weight, std=embed_dim**-0.5)
        self.layers = nn.ModuleList(
            Layer(embed_dim, hidden_w, blocks_per_layer, num_heads)
            for _ in range(n_layers)
        )
        self.norm_f = nn.LayerNorm(embed_dim)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
            ),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x) + self.pos_embed.weight[: x.shape[1]]
        for layer in self.layers:
            x = layer(x, self.causal_mask)
        return F.linear(self.norm_f(x), self.embed.weight)

    @property
    def device(self):
        return next(self.parameters()).device
