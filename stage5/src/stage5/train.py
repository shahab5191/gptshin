import math

import numpy as np
import torch
from torch import Tensor, nn, optim

from stage5.model import Model

LN2 = math.log(2)


def _ppl(loss: float) -> float:
    """exp(loss), capped so a huge early loss cannot overflow (exp(>709) does)."""
    return math.exp(min(loss, 700.0))


def load_tokens(path: str) -> np.memmap:
    """Map a flat uint16 token file into memory without reading it.

    The OS loads only the pages that are actually sliced, so a multi-GB file
    costs roughly one batch of RAM.
    """
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray, batch_size: int, block_size: int, device: str
) -> tuple[Tensor, Tensor]:
    """Sample `batch_size` windows at random start positions.

    x = data[i     : i + T]   input tokens
    y = data[i + 1 : i + T+1] the same window shifted by one: next-token targets

    Every start position is possible (like stride 1), but nothing is built up
    front - no window list, no shuffled index over billions of positions.
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    chunk = np.stack([data[i : i + block_size + 1] for i in ix]).astype(np.int64)
    xy = torch.from_numpy(chunk)
    if device == "cuda":
        xy = xy.pin_memory().to(device, non_blocking=True)
    else:
        xy = xy.to(device)
    return xy[:, :-1], xy[:, 1:]


def val_windows(data: np.ndarray, block_size: int) -> tuple[Tensor, Tensor]:
    """Fixed, non-overlapping windows - the same ones every evaluation."""
    n = (len(data) - 1) // block_size
    x = np.asarray(data[: n * block_size], dtype=np.int64).reshape(n, block_size)
    y = np.asarray(data[1 : n * block_size + 1], dtype=np.int64).reshape(n, block_size)
    return torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(
    model: Model,
    x: Tensor,
    y: Tensor,
    char_len: Tensor,
    batch_size: int = 64,
    max_windows: int | None = None,
) -> tuple[float, float, float]:
    """Return (loss per token, perplexity, bits per character).

    char_len[id] = number of characters token `id` decodes to. Bits per char
    divides the total loss by the characters the targets cover, so it stays
    comparable across tokenizers (per-token loss does not).
    """
    model.eval()
    if max_windows is not None:
        x, y = x[:max_windows], y[:max_windows]
    total_loss = 0.0
    total_tokens = 0
    total_chars = 0
    char_len = char_len.to(model.device)
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size].to(model.device)
        yb = y[i : i + batch_size].to(model.device)
        logits = model(xb)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), yb.reshape(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += yb.numel()
        total_chars += char_len[yb].sum().item()
    model.train()
    mean_loss = total_loss / total_tokens
    bpc = total_loss / total_chars / LN2
    return mean_loss, _ppl(mean_loss), bpc


def train(
    model: Model,
    data: np.ndarray,
    char_len: Tensor,
    val_x: Tensor | None = None,
    val_y: Tensor | None = None,
    steps=1000,
    lr=0.1,
    device: str | None = None,
    batch_size=64,
    log_every=1000,
    eval_batches: int | None = 200,
    weight_decay=0.01,
    save_path: str | None = None,
    save_every: int = 0,
    accum_steps: int = 4,
) -> tuple[list[float], list[float]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    char_len_dev = char_len.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    val_losses: list[float] = []
    running_loss = 0.0
    running_nats = 0.0
    running_chars = 0
    running_gn_max = 0.0
    micro_batch = batch_size // accum_steps
    for i in range(steps):
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(accum_steps):
            xb, yb = get_batch(data, micro_batch, model.block_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yhat = model(xb)
                loss = nn.functional.cross_entropy(
                    yhat.reshape(-1, yhat.shape[-1]), yb.reshape(-1)
                )
            (loss / accum_steps).backward()
            step_loss += loss.item() / accum_steps
            running_nats += loss.item() * yb.numel()
            running_chars += char_len_dev[yb].sum().item()
        grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        if not torch.math.isfinite(grad_norm):
            running_gn_max = max(running_gn_max, grad_norm.item())
            print(f"grad_norm max {running_gn_max:.2f}")
            optimizer.zero_grad()
            continue
        optimizer.step()
        losses.append(step_loss)
        running_loss += step_loss
        if save_path and save_every and (i + 1) % save_every == 0:
            torch.save(model.state_dict(), save_path)
            print(f"Model Stored at {save_path}")
        if (i + 1) % log_every == 0:
            mean_loss = running_loss / log_every
            bpc = running_nats / running_chars / LN2
            line = (
                f"step {i + 1:>7}/{steps}: "
                f"train loss {mean_loss:.4f} (ppl {_ppl(mean_loss):.1f}, "
                f"bpc {bpc:.3f})"
            )
            if val_x is not None and val_y is not None:
                val_loss, val_ppl, val_bpc = evaluate(
                    model,
                    val_x,
                    val_y,
                    char_len,
                    batch_size=micro_batch,
                    max_windows=eval_batches,
                )
                val_losses.append(val_loss)
                line += (
                    f"   val loss {val_loss:.4f} (ppl {val_ppl:.1f}, bpc {val_bpc:.3f})"
                )
            print(line, flush=True)
            running_loss = 0.0
            running_nats = 0.0
            running_chars = 0

    return (losses, val_losses)
