import os

import torch

from stage3 import tokenizer_v2 as tok
from stage3.model import Mlp
from stage3.train import evaluate, train

BLOCK_SIZE = 128
GEN_LEN = 300
DATA_FILE = "text8"
TOKEN_FILE = "tokens.txt"
MODEL_FILE = "model.pt"
MAX_CHARS = 100_000_000
EMBED_DIM = 512
STEPS = 150_000
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
    with open(DATA_FILE, "r", encoding="latin-1") as f:
        text = f.read(MAX_CHARS)
    if not text:
        raise Exception("Empty input file")

    tok.build_vocab(DATA_FILE, TOKEN_FILE)
    vocab_size = tok.vocab_size()
    print(f"{vocab_size=}")

    trn, val = tok.tokenize(text)
    x, y = tok.make_pairs(trn, BLOCK_SIZE)
    xv, yv = tok.make_pairs(val, BLOCK_SIZE, stride=BLOCK_SIZE)

    HIDDEN = [EMBED_DIM] * 3
    model = Mlp(BLOCK_SIZE, vocab_size, EMBED_DIM, HIDDEN)

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

    start = torch.randint(0, len(val) - BLOCK_SIZE, (1,)).item()
    context = val[start : start + BLOCK_SIZE].to(device)
    print(f"{context=}")
    generated = context.tolist()
    for _ in range(GEN_LEN):
        logits = model(context.unsqueeze(0))
        probs = logits.softmax(dim=1)
        nxt = torch.multinomial(probs, 1).item()
        generated.append(nxt)
        context = torch.tensor(generated[-BLOCK_SIZE:], device=device)
    print(tok.decode(generated))
