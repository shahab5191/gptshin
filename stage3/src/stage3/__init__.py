import torch

from stage3 import tokenizer as tok
from stage3.model import Mlp
from stage3.train import train

BLOCK_SIZE = 8
GEN_LEN = 300


def main() -> None:
    with open("input.txt", "r") as f:
        input = f.read()
    if not input:
        raise Exception("Empty input file")

    trn, val = tok.tokenize(input)
    vocab_size = tok.vocab_size()
    x, y = tok.make_pairs(trn, BLOCK_SIZE)

    model = Mlp(BLOCK_SIZE, vocab_size, [BLOCK_SIZE * vocab_size] * 2)

    train(model, x, y, steps=100000, lr=0.01)
    device = model.device

    xv, yv = tok.make_pairs(val, BLOCK_SIZE)
    xv, yv = xv.to(device), yv.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(xv)
        val_acc = (logits.argmax(dim=1) == yv).float().mean().item()
    print(f"val accuracy: {val_acc:.4f}")

    start = torch.randint(0, len(val) - BLOCK_SIZE, (1,)).item()
    context = val[start : start + BLOCK_SIZE].to(device)
    generated = context.tolist()
    for _ in range(GEN_LEN):
        logits = model(context.unsqueeze(0))
        probs = logits.softmax(dim=1)
        nxt = torch.multinomial(probs, 1).item()
        generated.append(nxt)
        context = torch.tensor(generated[-BLOCK_SIZE:], device=device)
    print(tok.decode(generated))
