import torch
from numpy.typing import NDArray
from torch import Tensor

from stage4.chained import Chained


def predict(model: Chained, x: Tensor) -> Tensor:
    return model(x).argmax(dim=1)


@torch.no_grad()
def generate(
    model: Chained,
    context: list[int] | Tensor | NDArray,
    n_tokens: int,
    temperature: float = 1.0,
) -> list[int]:
    tokens = [int(t) for t in context]
    if len(tokens) < model.block_size:
        raise ValueError(f"context has {len(tokens)} tokens, need {model.block_size}")
    model.eval()
    device = model.device
    for _ in range(n_tokens):
        window = torch.tensor(tokens[-model.block_size :], device=device)
        logits = model(window.unsqueeze(0))
        probs = (logits / temperature).softmax(dim=1)
        tokens.append(int(torch.multinomial(probs, 1).item()))
    return tokens[len(context) :]
