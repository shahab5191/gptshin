import torch
from torch import Tensor, nn, tanh, zeros


class Chained(nn.Module):
    offsets: Tensor  # buffer: 0..tokens_per_frame-1

    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        embed_dim: int,
        tokens_per_frame: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.tokens_per_frame = tokens_per_frame
        self.hidden_dim = hidden_dim
        self.n_frames = block_size // tokens_per_frame

        self.embed = nn.Embedding(vocab_size, embed_dim)
        # where each input patch sits in its frame (0..tokens_per_frame-1)
        self.pos_embed = nn.Embedding(tokens_per_frame, embed_dim)
        # which position the predicted (next) patch is at
        self.target_embed = nn.Embedding(tokens_per_frame, hidden_dim)
        # small init so it doesn't swamp h (tanh output, std ~0.5) at the start
        nn.init.normal_(self.target_embed.weight, std=0.02)
        self.register_buffer("offsets", torch.arange(tokens_per_frame), persistent=False)

        self.frame_net = nn.Linear(tokens_per_frame * embed_dim, hidden_dim)
        self.bridge = nn.Linear(hidden_dim, hidden_dim)
        self.main_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor, phase: Tensor) -> Tensor:
        # x: (B, block_size) int patch ids
        # phase: (B,) in-frame position of x[:, 0]; token i sits at (phase + i) % tpf
        tpf = self.tokens_per_frame
        e = self.embed(x).view(-1, self.n_frames, tpf, self.embed_dim)
        # every tpf-token chunk covers each frame position once, in the same rotated
        # order, so look positions up once per window and reuse them for all chunks
        pos = (phase.unsqueeze(1) + self.offsets) % tpf  # (B, tpf)
        e = e + self.pos_embed(pos).unsqueeze(1)  # (B, n_frames, tpf, embed_dim)
        frames = e.flatten(2).transpose(0, 1)  # (n_frames, B, tpf * embed_dim)
        h = zeros(frames.shape[1], self.hidden_dim, device=x.device)
        for f in frames:
            h = tanh(self.frame_net(f) + self.bridge(h))
        # target is block_size tokens after x[:, 0]; block_size is a whole number of
        # frames, so its position is phase itself
        target_pos = (phase + self.block_size) % self.tokens_per_frame
        return self.main_net(h + self.target_embed(target_pos))  # (B, out_dim)

    @property
    def device(self):
        return next(self.parameters()).device
