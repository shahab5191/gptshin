"""A deeper version of Chained: same idea, three upgrades.

Chained:  frame -> 1 Linear -> 1 RNN layer -> 3-layer ReLU head
Deep:     frame -> Linear + residual blocks -> stacked RNN layers -> residual-block head

1. Frame encoder: after the first Linear, a few residual blocks let the model
   build up features of a frame (edges, the arm, objects) before summarising it.
2. Stacked RNN: several recurrent layers, each with LayerNorm, so the memory
   of past frames is processed in more than one step.
3. Head made of residual blocks instead of a plain Linear/ReLU stack.

Residual block = x + MLP(LayerNorm(x)). The "x +" is a shortcut: each block
only has to learn a correction to its input, and gradients flow straight down
the shortcuts, so deep stacks keep training. LayerNorm rescales each vector to
mean 0 / std 1, so values stay in a sane range however deep the stack gets.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult)
        self.fc2 = nn.Linear(dim * mult, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class RNNLayer(nn.Module):
    """h_t = tanh(LayerNorm(W_in x_t + W_rec h_{t-1})) - Chained's RNN, plus LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.inp = nn.Linear(dim, dim)
        self.rec = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq: Tensor) -> Tensor:
        # seq: (n_frames, B, dim) -> hidden state after every frame, same shape
        inputs = self.inp(seq)  # all frames at once; only the recurrence is a loop
        h = torch.zeros_like(seq[0])
        out = []
        for x in inputs:
            h = torch.tanh(self.norm(x + self.rec(h)))
            out.append(h)
        return torch.stack(out)


class Deep(nn.Module):
    offsets: Tensor  # buffer: 0..tokens_per_frame-1

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        embed_dim: int,
        tokens_per_frame: int,
        hidden_dim: int,
        out_dim: int,
        frame_blocks: int = 2,
        rnn_layers: int = 2,
        head_blocks: int = 4,
    ):
        super().__init__()
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.tokens_per_frame = tokens_per_frame
        self.hidden_dim = hidden_dim
        self.n_frames = block_size // tokens_per_frame

        # same embeddings as Chained
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(tokens_per_frame, embed_dim)
        self.target_embed = nn.Embedding(tokens_per_frame, hidden_dim)
        nn.init.normal_(self.target_embed.weight, std=0.02)
        self.register_buffer("offsets", torch.arange(tokens_per_frame), persistent=False)

        self.frame_in = nn.Linear(tokens_per_frame * embed_dim, hidden_dim)
        self.frame_net = nn.Sequential(*(ResBlock(hidden_dim) for _ in range(frame_blocks)))
        self.rnn = nn.ModuleList(RNNLayer(hidden_dim) for _ in range(rnn_layers))
        self.head = nn.Sequential(
            *(ResBlock(hidden_dim) for _ in range(head_blocks)),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor, phase: Tensor) -> Tensor:
        # x: (B, block_size) patch ids; phase: (B,) in-frame position of x[:, 0]
        tpf = self.tokens_per_frame
        e = self.embed(x).view(-1, self.n_frames, tpf, self.embed_dim)
        pos = (phase.unsqueeze(1) + self.offsets) % tpf  # (B, tpf)
        e = e + self.pos_embed(pos).unsqueeze(1)
        # encode every frame at once: (B, n_frames, hidden)
        f = self.frame_net(self.frame_in(e.flatten(2)))
        seq = f.transpose(0, 1)  # (n_frames, B, hidden): time first for the loop
        for layer in self.rnn:
            seq = seq + layer(seq)  # residual around each RNN layer too
        h = seq[-1]  # state after the newest frame
        target_pos = (phase + self.block_size) % tpf
        return self.head(h + self.target_embed(target_pos))  # (B, out_dim)

    @property
    def device(self):
        return next(self.parameters()).device
