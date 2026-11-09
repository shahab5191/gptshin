"""Animate a single picture with the trained model.

The picture is resized/cropped the same way training frames were, encoded into
patch tokens with the codebook, and repeated CONTEXT_FRAMES times as the starting
context. The model then generates the following frames one token at a time.

    uv run stage4-animate photo.jpg
    uv run stage4-animate photo.jpg --frames 40 --temperature 0.7 --out walk.gif
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from stage4 import (
    CODEBOOK_FILE,
    CONTEXT_FRAMES,
    FPS,
    GEN_FRAMES,
    HIDDEN,
    MODEL_FILE,
    SIZE,
    TEMPERATURE,
    TOKENS_PER_FRAME,
    build_model,
)
from stage4 import tokenizer as tok
from stage4.models import Model
from stage4.predict import generate
from stage4.utils import save_gif


def load_picture(path: Path, size: int = SIZE) -> NDArray:
    """Picture -> (size, size, 3) uint8, matching video_data's ffmpeg filters:
    scale so the short side fits (area filter), then center-crop. No stretching."""
    img = Image.open(path).convert("RGB")
    scale = size / min(img.size)
    w, h = max(size, round(img.width * scale)), max(size, round(img.height * scale))
    img = img.resize((w, h), Image.Resampling.BOX)
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)))


def load_model(path: str | Path, vocab_size: int, device: str) -> Model:
    model = build_model(vocab_size)  # same model stage4.main() trains
    state = torch.load(path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise SystemExit(
            f"{path} doesn't match the current config in stage4/__init__.py "
            f"(SIZE={SIZE}, HIDDEN={HIDDEN}, vocab={vocab_size}). "
            f"Was it trained with different settings?\n{e}"
        ) from None
    return model.to(device)


def animate(
    picture: NDArray,
    model: Model,
    codebook: NDArray,
    n_frames: int = GEN_FRAMES,
    temperature: float = TEMPERATURE,
) -> NDArray:
    """Returns (1 + n_frames, SIZE, SIZE, 3): the tokenized picture, then generated frames."""
    ids = tok.encode(picture[None], codebook)  # (TOKENS_PER_FRAME,)
    context = np.tile(ids, CONTEXT_FRAMES)  # static picture as the "past" frames
    new_ids = generate(model, context, n_frames * TOKENS_PER_FRAME, temperature)
    all_ids = np.concatenate([ids, np.asarray(new_ids, dtype=ids.dtype)])
    return tok.decode_frames(all_ids, codebook, SIZE, SIZE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a picture with the stage4 model")
    parser.add_argument("image", type=Path, help="input picture (jpg, png, ...)")
    parser.add_argument("--frames", type=int, default=GEN_FRAMES, help="frames to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--model", type=Path, default=Path(MODEL_FILE))
    parser.add_argument("--codebook", type=Path, default=CODEBOOK_FILE)
    parser.add_argument("--out", type=Path, help="output gif (default: <image>_anim.gif)")
    parser.add_argument("--seed", type=int, help="random seed for repeatable output")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or args.image.with_name(f"{args.image.stem}_anim.gif")

    codebook = np.load(args.codebook)
    model = load_model(args.model, len(codebook), device)
    picture = load_picture(args.image)
    print(f"animating {args.image} ({SIZE}x{SIZE}) for {args.frames} frames on {device}")
    frames = animate(picture, model, codebook, args.frames, args.temperature)
    save_gif(frames, out, duration=1000 // FPS)  # real-time playback
    print(f"saved {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
