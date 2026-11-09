from pathlib import Path

import numpy as np
import torch
from numpy._typing import NDArray
from torch import Tensor

from stage4 import tokenizer

VAL_STRIDE = 7


def make_pairs(data: Tensor, block_size: int, stride: int = 1) -> tuple[Tensor, Tensor]:
    n_windows = (len(data) - block_size) // stride
    x = data.unfold(0, block_size, stride)[:n_windows]
    y = data[block_size : block_size + n_windows * stride : stride]
    return x, y


class Windows:
    """Training windows cut out of one token stream on demand.

    Behaves like the (N, block_size) tensor make_pairs would return (len, .shape,
    X[idx], X[a:b]), but only stores the tokens once plus one start index per
    window. Materializing every stride-1 window would copy each token
    block_size times.
    """

    def __init__(self, tokens: Tensor, starts: Tensor, block_size: int):
        self.tokens = tokens
        self.starts = starts
        self.offsets = torch.arange(block_size)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.starts), len(self.offsets))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx) -> Tensor:
        return self.tokens[self.starts[idx].unsqueeze(-1) + self.offsets]


def build_dataset(
    frames_dir: Path,
    centroids: NDArray,
    block_size: int,
    tokens_per_frame: int,
    max_clips: int | None = None,
    train_frac: float = 0.9,
) -> tuple[Windows, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Returns (x, y, phase, xv, yv, phase_v).

    phase[i] is the in-frame position (0..tokens_per_frame-1) of window i's first
    token; the model needs it for its position embeddings.
    """
    streams: list[Tensor] = []
    starts: list[Tensor] = []
    offset = 0
    xvs: list[Tensor] = []
    yvs: list[Tensor] = []
    pvs: list[Tensor] = []

    npy_files = sorted(frames_dir.glob("*.npy"))
    if max_clips is not None:
        npy_files = npy_files[:max_clips]

    print(f"building dataset: encoding {len(npy_files)} clips")
    for i, fn in enumerate(npy_files):
        clip = np.load(fn)
        codes = tokenizer.encode(clip, centroids)  # (T * tokens_per_frame,) int16
        # split on a frame boundary so val windows start at the top-left patch
        n = int(train_frac * len(codes)) // tokens_per_frame * tokens_per_frame
        trn = torch.from_numpy(codes[:n])
        val = torch.from_numpy(codes[n:])
        if len(trn) <= block_size or len(val) <= block_size:
            print(f"  [{i + 1}/{len(npy_files)}] {fn.name}: skipped (too short)")
            continue
        # every stride-1 window inside this clip (never crossing into the next clip)
        streams.append(trn)
        starts.append(torch.arange(len(trn) - block_size) + offset)
        offset += len(trn)
        # val target position in the frame is (start + block_size) % tokens_per_frame.
        # A stride that's a multiple of tokens_per_frame (e.g. block_size) would
        # always target the same patch; a small odd stride (coprime with 256)
        # spreads targets over the whole frame. xv[0] still starts on a frame
        # boundary, so generation from it stays aligned.
        xv, yv = make_pairs(val, block_size, stride=VAL_STRIDE)
        xvs.append(xv)
        yvs.append(yv)
        # val starts on a frame boundary, window k starts k * VAL_STRIDE later
        pvs.append(torch.arange(len(xv)) * VAL_STRIDE % tokens_per_frame)
        if (i + 1) % 50 == 0 or i + 1 == len(npy_files):
            print(f"  [{i + 1}/{len(npy_files)}] clips encoded")

    tokens = torch.cat(streams)
    start_idx = torch.cat(starts)
    x = Windows(tokens, start_idx, block_size)
    y = tokens[start_idx + block_size]
    # every clip's train stream starts on a frame boundary and is a whole number of
    # frames long, so a window's in-frame position is just its global start index
    phase = start_idx % tokens_per_frame
    return x, y, phase, torch.cat(xvs), torch.cat(yvs), torch.cat(pvs)


def build_clip_dataset(
    train_clips: NDArray,
    val_clips: NDArray,
    block_size: int,
    tokens_per_frame: int,
    val_stride: int = VAL_STRIDE,
) -> tuple[Windows, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Like build_dataset, for a dataset that already has separate train and
    val clips of equal length (BAIR: (N, 30 * 256) token ids per split).

    Same return values. Windows never cross from one clip into the next.
    """
    n, length = train_clips.shape
    per_clip = length - block_size  # stride-1 windows inside one clip
    tokens = torch.from_numpy(np.ascontiguousarray(train_clips)).view(-1)
    # int32 keeps 221M start indices at 0.9 GB instead of 1.8 GB
    clip_start = torch.arange(n, dtype=torch.int32) * length
    offset = torch.arange(per_clip, dtype=torch.int32)
    starts = (clip_start[:, None] + offset[None, :]).view(-1)
    x = Windows(tokens, starts, block_size)
    y = tokens[starts + block_size]
    # clips start on a frame boundary, so the in-frame position only depends on
    # how far into its clip a window starts
    phase = offset.repeat(n) % tokens_per_frame

    xvs, yvs, pvs = [], [], []
    for clip in val_clips:
        xv, yv = make_pairs(torch.from_numpy(clip), block_size, stride=val_stride)
        xvs.append(xv)
        yvs.append(yv)
        pvs.append(torch.arange(len(xv)) * val_stride % tokens_per_frame)
    print(f"dataset: {len(x):,} train windows, {sum(map(len, xvs)):,} val windows")
    return x, y, phase, torch.cat(xvs), torch.cat(yvs), torch.cat(pvs)
