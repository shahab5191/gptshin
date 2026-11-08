from pathlib import Path

import numpy as np
from numpy._typing import NDArray

patches = []


def _extract_patches(clip: NDArray) -> NDArray:
    T, H, W, C = clip.shape
    return clip.reshape(T, 8, 4, 8, 4, 3).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 48)


def list_patches(frames_dir: Path) -> NDArray:
    all_patches: list[NDArray] = []
    npy_files = [file.name for file in frames_dir.iterdir() if file.suffix == ".npy"]
    for fn in npy_files:
        clip: NDArray = np.load(frames_dir / fn)
        all_patches.append(_extract_patches(clip))
    patches = np.concatenate(all_patches, axis=0)
    return patches


def extract_codebook(patches: NDArray, iters=20, chunk=8192):
    rng = np.random.default_rng(100)
    n = patches.shape[0]
    k = 1024
    patches = patches.astype(np.float32) / 255.0
    centroids = patches[rng.choice(n, k, replace=False)]

    for _ in range(iters):
        labels = np.empty(n, dtype=np.int32)
        for s in range(0, n, chunk):
            p = patches[s : s + chunk]
            d = ((p[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            labels[s : s + chunk] = d.argmin(1)
        sums = np.zeros((k, 48), dtype=np.float64)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(sums, labels, patches)
        np.add.at(counts, labels, 1)
        empty = counts == 0
        centroids = sums / np.maximum(counts, 1)[:, None]
        centroids[empty] = patches[rng.choice(n, empty.sum(), replace=False)]
        centroids = centroids.astype(np.float32)
    return centroids


def tokenize(frame_dir: Path):
    patches = list_patches(frame_dir)
