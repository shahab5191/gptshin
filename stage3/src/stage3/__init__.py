import torch

from stage3 import tokenizer_v2 as tok
from stage3.model import Mlp
from stage3.train import train

BLOCK_SIZE = 32
GEN_LEN = 300
DATA_FILE = "text8"
TOKEN_FILE = "tokens.txt"
MAX_CHARS = 5_000_000
EMBED_DIM = 1024


def _accuracy(model: Mlp, x: torch.Tensor, y: torch.Tensor, batch_size=4096) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = x[i: i + batch_size].to(model.device)
            yb = y[i: i + batch_size].to(model.device)
            logits = model(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.shape[0]
    return correct / total


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

    HIDDEN = [EMBED_DIM] * 3
    model = Mlp(BLOCK_SIZE, vocab_size, EMBED_DIM, HIDDEN)

    train(model, x, y, steps=100000, lr=1e-3)
    device = model.device

    xv, yv = tok.make_pairs(val, BLOCK_SIZE)
    val_acc = _accuracy(model, xv, yv)
    print(f"val accuracy: {val_acc:.4f}")

    start = torch.randint(0, len(val) - BLOCK_SIZE, (1,)).item()
    context = val[start: start + BLOCK_SIZE].to(device)
    print(f"{context=}")
    generated = context.tolist()
    for _ in range(GEN_LEN):
        logits = model(context.unsqueeze(0))
        probs = logits.softmax(dim=1)
        nxt = torch.multinomial(probs, 1).item()
        generated.append(nxt)
        context = torch.tensor(generated[-BLOCK_SIZE:], device=device)
    print(tok.decode(generated))
