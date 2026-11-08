import os
from pathlib import Path

import numpy as np
import torch

from stage4.data import build_dataset
from stage4.model import Mlp
from stage4.train import evaluate, train
from stage4.utils import save_gif

BLOCK_SIZE = 192
GEN_LEN = 128
EMBED_DIM = 512
STEPS = 200_000
MODEL_FILE = "model.pt"
LEARNING_RATE = 1e-3
SAVE_EVERY = 10_000


def _load_compatible(model: Mlp, path: str) -> bool:
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    current = model.state_dict()
    if set(current) != set(state) or any(
        current[k].shape != state[k].shape for k in current
    ):
        print("checkpoint architecture changed - starting fresh")
        return False
    model.load_state_dict(state)
    print(f"loaded model from {path}")
    return True


def main() -> None:
    from stage4 import tokenizer as tok

    codebook = tok.tokenize(Path("frames"))
    x, y, xv, yv = build_dataset(Path("frames"), codebook, BLOCK_SIZE, 50, 0.9)

    HIDDEN = [256] * 3
    model = Mlp(BLOCK_SIZE, tok.vocab_size(), EMBED_DIM, HIDDEN)

    _load_compatible(model, MODEL_FILE)

    train(
        model,
        x,
        y,
        val_x=xv,
        val_y=yv,
        steps=STEPS,
        lr=LEARNING_RATE,
        save_path=MODEL_FILE,
        save_every=SAVE_EVERY,
    )
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"saved model to {MODEL_FILE}")

    device = model.device

    val_loss, val_ppl = evaluate(model, xv, yv)
    print(f"final val loss {val_loss:.4f} (ppl {val_ppl:.1f})")

    context = xv[0].to(device).long()
    print(f"{context=}")
    generated = context.tolist()
    for _ in range(GEN_LEN):
        logits = model(context.unsqueeze(0))
        probs = logits.softmax(dim=1)
        nxt = torch.multinomial(probs, 1).item()
        generated.append(nxt)
        context = torch.tensor(generated[-BLOCK_SIZE:], device=device)

    gen_ids = generated[BLOCK_SIZE:]
    patches = codebook[gen_ids]
    frames = (patches.reshape(-1, 32, 32, 3) * 255).astype(np.uint8)
    save_gif(frames, "gen.gif")
