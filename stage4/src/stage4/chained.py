from torch import Tensor, nn, tanh, zeros


class Chained(nn.Module):
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
        self.frame_net = nn.Linear(tokens_per_frame * embed_dim, hidden_dim)
        self.bridge = nn.Linear(hidden_dim, hidden_dim)
        self.main_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, block_size) int patch ids
        e = self.embed(x)  # (B, block_size, embed_dim)
        e = e.reshape(-1, self.n_frames, self.tokens_per_frame * self.embed_dim)
        frames = e.transpose(0, 1)  # (n_frames, B, frame_dim)
        h = zeros(frames.shape[1], self.hidden_dim, device=x.device)
        for f in frames:
            h = tanh(self.frame_net(f) + self.bridge(h))
        return self.main_net(h)  # (B, out_dim)

    @property
    def device(self):
        return next(self.parameters()).device
