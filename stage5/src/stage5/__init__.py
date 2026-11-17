import os

import torch

from stage5.bpe import BPE
from stage5.model import Model
from stage5.train import evaluate, load_tokens, train, val_windows

BLOCK_SIZE = 8192
GEN_LEN = 300
TRAIN_FILE = "data/owt_bpe_train.bin"
VAL_FILE = "data/owt_bpe_val.bin"
TEXT8_VAL_FILE = "data/text8_bpe_val.bin"  # same text as the old text8 runs' val
MODEL_FILE = "model_bpe.pt"  # new name, so the text8 model.pt is not overwritten
EMBED_DIM = 512
STEPS = 15_000  # ~1 pass over ~2B tokens at 131k tokens/step
LEARNING_RATE = 1e-4
SAVE_EVERY = 1_000
LOG_EVERY = 250
HIDDEN_W = 2048  # 4x EMBED_DIM, the transformer ratio
N_LAYERS = 8  # repeated [Attention -> MLP] layers
BLOCKS_PER_LAYER = 1  # residual blocks inside each layer's MLP
BATCH_SIZE = 16  # effective batch per optimizer step (16 x 8192 = 131k tokens)
ACCUM_STEPS = 16  # micro-batch = 16 // 16 = 1 window


def _load_compatible(model: Model, path: str) -> bool:
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
    bpe = BPE.load()
    vocab_size = bpe.vocab_size
    print(f"{vocab_size=}")
    # characters each token decodes to - turns loss into bits per character
    char_len = torch.tensor([len(b) for b in bpe.vocab])

    data = load_tokens(TRAIN_FILE)
    xv, yv = val_windows(load_tokens(VAL_FILE), BLOCK_SIZE)
    xt, yt = val_windows(load_tokens(TEXT8_VAL_FILE), BLOCK_SIZE)
    print(f"train {len(data):,} tokens, val {xv.numel():,}, text8 val {xt.numel():,}")

    model = Model(
        BLOCK_SIZE,
        vocab_size,
        EMBED_DIM,
        HIDDEN_W,
        BLOCKS_PER_LAYER,
        N_LAYERS,
    )

    _load_compatible(model, MODEL_FILE)

    train(
        model,
        data,
        char_len,
        val_x=xv,
        val_y=yv,
        steps=STEPS,
        lr=LEARNING_RATE,
        save_path=MODEL_FILE,
        save_every=SAVE_EVERY,
        log_every=LOG_EVERY,
        batch_size=BATCH_SIZE,
        accum_steps=ACCUM_STEPS,
    )
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"saved model to {MODEL_FILE}")
    device = model.device

    for name, x, y in (("owt val", xv, yv), ("text8 val", xt, yt)):
        loss, ppl, bpc = evaluate(
            model, x, y, char_len, batch_size=BATCH_SIZE // ACCUM_STEPS
        )
        print(f"final {name:<9} loss {loss:.4f} (ppl {ppl:.1f})  bpc {bpc:.3f}")

    val = load_tokens(VAL_FILE)
    start = torch.randint(0, len(val) - BLOCK_SIZE, (1,)).item()
    context = torch.from_numpy(val[start : start + BLOCK_SIZE].astype("int64")).to(
        device
    )
    generated = context.tolist()
    print(f"prompt: ...{bpe.decode(generated[-60:])}")
    for _ in range(GEN_LEN):
        logits = model(context.unsqueeze(0))
        probs = logits[0, -1].softmax(dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        generated.append(nxt)
        context = torch.tensor(generated[-BLOCK_SIZE:], device=device)
    print(f"generated: {bpe.decode(generated[-GEN_LEN:])}")
