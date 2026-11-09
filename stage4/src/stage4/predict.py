import torch
from numpy.typing import NDArray
from torch import Tensor

from stage4.models import Model


def predict(model: Model, x: Tensor, phase: Tensor) -> Tensor:
    return model(x, phase).argmax(dim=1)


@torch.no_grad()
def generate(
    model: Model,
    context: list[int] | Tensor | NDArray,
    n_tokens: int,
    temperature: float = 1.0,
    phase: int = 0,
) -> Tensor:
    """Sample n_tokens after `context`; returns only the new tokens (on the CPU).

    context is one sequence (L,) or a batch of sequences (B, L) that are
    generated side by side - B clips cost about the same time as one.
    phase is the in-frame position of context[..., 0] (0 = it starts on a frame
    boundary, which is the case for val clips and for animate's picture).
    """
    ctx = torch.as_tensor(context).long()
    single = ctx.dim() == 1
    if single:
        ctx = ctx.unsqueeze(0)
    if ctx.shape[1] < model.block_size:
        raise ValueError(f"context has {ctx.shape[1]} tokens, need {model.block_size}")
    model.eval()
    device = model.device
    tokens = torch.cat([ctx, ctx.new_zeros(len(ctx), n_tokens)], dim=1).to(device)
    for i in range(ctx.shape[1], tokens.shape[1]):
        window = tokens[:, i - model.block_size : i]
        # the window starts i - block_size tokens after context[0]
        start = (phase + i - model.block_size) % model.tokens_per_frame
        starts = torch.full((len(tokens),), start, device=device)
        logits = model(window, starts)
        probs = (logits / temperature).softmax(dim=1)
        tokens[:, i] = torch.multinomial(probs, 1).squeeze(1)
    new = tokens[:, ctx.shape[1] :].cpu()
    return new[0] if single else new
