import os
from pathlib import Path

import torch

from stage4.data import build_dataset
from stage4.model import Mlp
from stage4.train import evaluate, train
from stage4.utils import save_gif
from stage4.video_data import load_videos

SIZE = 64  # frames are SIZE x SIZE pixels
FPS = 5
TOKENS_PER_FRAME = (SIZE // 4) ** 2  # 4x4-pixel patches -> 256 tokens at 64x64
CONTEXT_FRAMES = 3
BLOCK_SIZE = TOKENS_PER_FRAME * CONTEXT_FRAMES
GEN_FRAMES = 30
GEN_LEN = TOKENS_PER_FRAME * GEN_FRAMES
TEMPERATURE = 0.8
FRAMES_DIR = Path("frames") / f"{SIZE}x{SIZE}_fps{FPS}"
CODEBOOK_FILE = Path("frames") / "codebooks" / f"{SIZE}x{SIZE}_fps{FPS}.npy"
MAX_CLIPS = None  # all clips
EMBED_DIM = 64
STEPS = 20_000
MODEL_FILE = "model.pt"
LEARNING_RATE = 3e-4


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

    # extracts once, then cached
    load_videos(size=(SIZE, SIZE), fps=FPS, frames_dir=str(FRAMES_DIR))
    # codebook sampled from patches of all clips (k-means on a 1M-patch subset)
    codebook = tok.tokenize(FRAMES_DIR, codebook_file=CODEBOOK_FILE, max_clips=None)
    x, y, xv, yv = build_dataset(
        FRAMES_DIR, codebook, BLOCK_SIZE, TOKENS_PER_FRAME, MAX_CLIPS, 0.9
    )

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
    )
    # train() saved the best-val checkpoint; use it rather than the last weights
    model.load_state_dict(
        torch.load(MODEL_FILE, map_location=model.device, weights_only=True)
    )
    print(f"loaded best model from {MODEL_FILE}")

    device = model.device

    val_loss, val_ppl = evaluate(model, xv, yv)
    print(f"final val loss {val_loss:.4f} (ppl {val_ppl:.1f})")

    model.eval()  # evaluate() leaves the model in train mode (dropout on)
    context = xv[0].to(device).long()
    print(f"{context=}")
    generated = context.tolist()
    with torch.no_grad():
        for _ in range(GEN_LEN):
            logits = model(context.unsqueeze(0))
            probs = (logits / TEMPERATURE).softmax(dim=1)
            nxt = torch.multinomial(probs, 1).item()
            generated.append(nxt)
            context = torch.tensor(generated[-BLOCK_SIZE:], device=device)

    gen_ids = generated[BLOCK_SIZE:]
    frames = tok.decode_frames(gen_ids, codebook, SIZE, SIZE)
    save_gif(frames, "gen.gif")
