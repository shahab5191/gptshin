import os
from pathlib import Path

import numpy as np
import torch

from stage4.deep import Deep
from stage4.predict import generate
from stage4.train import evaluate, train
from stage4.utils import save_gif

DATASET = "bair"  # "bair" (robot pushing, colour) or "kth" (people, greyscale)
SIZE = 64  # frames are SIZE x SIZE pixels
TOKENS_PER_FRAME = (SIZE // 4) ** 2  # 4x4-pixel patches -> 256 tokens at 64x64

if DATASET == "bair":
    FPS = 10  # BAIR was recorded at ~10 Hz; only sets gif playback speed
    CONTEXT_FRAMES = 10
    GEN_FRAMES = 20  # BAIR clips are 30 frames: 10 context + 20 to predict
    VOCAB_SIZE = 4096  # colour patches vary more than grey ones
    CODEBOOK_FILE = Path("frames") / "codebooks" / f"bair_{VOCAB_SIZE}.npy"
    VAL_STRIDE = 31  # coprime with 256; 256 test clips -> ~42k val windows
else:
    FPS = 12
    CONTEXT_FRAMES = 15
    GEN_FRAMES = 150
    VOCAB_SIZE = 512
    CODEBOOK_FILE = Path("frames") / "codebooks" / f"{SIZE}x{SIZE}_fps{FPS}.npy"
    VAL_STRIDE = 7
FRAMES_DIR = Path("frames") / f"{SIZE}x{SIZE}_fps{FPS}"  # KTH only

BLOCK_SIZE = TOKENS_PER_FRAME * CONTEXT_FRAMES
GEN_LEN = TOKENS_PER_FRAME * GEN_FRAMES
TEMPERATURE = 0.6
MAX_CLIPS = None  # all clips (KTH)
EMBED_DIM = 64
HIDDEN = 1024  # width of every layer after the embeddings
FRAME_BLOCKS = 2  # residual blocks per frame before the RNN
RNN_LAYERS = 2
HEAD_BLOCKS = 4  # residual blocks in the head
STEPS = 200_000
MODEL_FILE = "model.pt"
LEARNING_RATE = 3e-4
N_SAMPLES = 4  # val clips generated side by side for gen.gif


def build_model(vocab_size: int) -> Deep:
    return Deep(
        BLOCK_SIZE,
        vocab_size,
        EMBED_DIM,
        TOKENS_PER_FRAME,
        HIDDEN,
        vocab_size,
        frame_blocks=FRAME_BLOCKS,
        rnn_layers=RNN_LAYERS,
        head_blocks=HEAD_BLOCKS,
    )


def _load_compatible(model: torch.nn.Module, path: str) -> bool:
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


def _load_data():
    """Returns (codebook, (x, y, phase, xv, yv, phase_v))."""
    from stage4 import tokenizer as tok

    if DATASET == "bair":
        from stage4 import bair
        from stage4.data import build_clip_dataset

        bair.convert()  # one pass over the tar, then cached in frames/bair/
        codebook = bair.codebook(CODEBOOK_FILE, VOCAB_SIZE)
        train_ids = bair.tokens("train", codebook, CODEBOOK_FILE)
        val_ids = bair.tokens("test", codebook, CODEBOOK_FILE)
        data = build_clip_dataset(
            train_ids, val_ids, BLOCK_SIZE, TOKENS_PER_FRAME, VAL_STRIDE
        )
        return codebook, data

    from stage4.data import build_dataset
    from stage4.video_data import load_videos

    load_videos(size=(SIZE, SIZE), fps=FPS, frames_dir=str(FRAMES_DIR))
    codebook = tok.tokenize(FRAMES_DIR, codebook_file=CODEBOOK_FILE, max_clips=None)
    data = build_dataset(FRAMES_DIR, codebook, BLOCK_SIZE, TOKENS_PER_FRAME, MAX_CLIPS, 0.9)
    return codebook, data


def _sample_gif(model: Deep, codebook: np.ndarray, xv, pv) -> None:
    """gen.gif: N_SAMPLES val clips stacked vertically, real (left) vs generated (right).

    Both sides start from the same CONTEXT_FRAMES real frames; after that the
    left keeps playing the real clip and the right is the model's guess.
    """
    from stage4 import tokenizer as tok

    def frames(ids) -> np.ndarray:
        return tok.decode_frames(ids, codebook, SIZE, SIZE)

    if DATASET != "bair":  # KTH: one generated clip, no ground truth to compare
        gen = generate(model, xv[0], GEN_LEN, TEMPERATURE, phase=int(pv[0]))
        save_gif(frames(gen), "gen.gif", duration=1000 // FPS)
        return

    from stage4 import bair

    real = bair.tokens("test", codebook, CODEBOOK_FILE)[:N_SAMPLES]  # (N, 30 * 256)
    context = torch.from_numpy(real[:, :BLOCK_SIZE].astype(np.int64))
    gen = generate(model, context, GEN_LEN, TEMPERATURE).numpy()  # clips start at phase 0
    fake = np.concatenate([real[:, :BLOCK_SIZE], gen], axis=1)
    rows = [np.concatenate([frames(r), frames(f)], axis=2) for r, f in zip(real, fake)]
    save_gif(np.concatenate(rows, axis=1), "gen.gif", duration=1000 // FPS)
    print(f"saved gen.gif: {N_SAMPLES} test clips, real (left) vs generated (right)")


def main() -> None:
    codebook, (x, y, p, xv, yv, pv) = _load_data()

    model = build_model(len(codebook))
    n_params = sum(t.numel() for t in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M parameters")
    _load_compatible(model, MODEL_FILE)

    train(
        model,
        x,
        y,
        p,
        val_x=xv,
        val_y=yv,
        val_phase=pv,
        steps=STEPS,
        lr=LEARNING_RATE,
        save_path=MODEL_FILE,
    )
    # train() saved the best-val checkpoint; use it rather than the last weights
    model.load_state_dict(
        torch.load(MODEL_FILE, map_location=model.device, weights_only=True)
    )
    print(f"loaded best model from {MODEL_FILE}")

    val_loss, val_ppl = evaluate(model, xv, yv, pv)
    print(f"final val loss {val_loss:.4f} (ppl {val_ppl:.1f})")
    _sample_gif(model, codebook, xv, pv)
