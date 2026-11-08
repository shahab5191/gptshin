from pathlib import Path

import numpy as np
import torch
from numpy._typing import NDArray
from torch import Tensor

from stage4 import tokenizer


def make_pairs(data: Tensor, block_size: int, stride: int = 1) -> tuple[Tensor, Tensor]:
    n_windows = (len(data) - block_size) // stride
    x = data.unfold(0, block_size, stride)[:n_windows]
    y = data[block_size : block_size + n_windows * stride : stride]
    return x, y


def build_dataset(
    frames_dir: Path,
    centroids: NDArray,
    block_size: int,
    max_clips: int | None = None,
    train_frac: float = 0.9,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    xs: list[Tensor] = []
    ys: list[Tensor] = []
    xvs: list[Tensor] = []
    yvs: list[Tensor] = []

    npy_files = sorted(frames_dir.glob("*.npy"))
    if max_clips is not None:
        npy_files = npy_files[:max_clips]

    print(f"building dataset: encoding {len(npy_files)} clips")
    for i, fn in enumerate(npy_files):
        clip = np.load(fn)
        codes = tokenizer.encode(clip, centroids)  # (T*64,) int16
        n = int(train_frac * len(codes))
        trn = torch.from_numpy(codes[:n])
        val = torch.from_numpy(codes[n:])
        if len(trn) <= block_size or len(val) <= block_size:
            print(f"  [{i + 1}/{len(npy_files)}] {fn.name}: skipped (too short)")
            continue
        x, y = make_pairs(trn, block_size, stride=1)
        xv, yv = make_pairs(val, block_size, stride=block_size)
        xs.append(x)
        ys.append(y)
        xvs.append(xv)
        yvs.append(yv)
        print(f"  [{i + 1}/{len(npy_files)}] {fn.name}: {len(codes)} codes")

    return torch.cat(xs), torch.cat(ys), torch.cat(xvs), torch.cat(yvs)
