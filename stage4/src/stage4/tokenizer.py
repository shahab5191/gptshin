from pathlib import Path

import numpy as np
from numpy._typing import NDArray

CODEBOOK_FILE = "codebook.npy"
VOCAB_SIZE = 512


def _extract_patches(clip: NDArray) -> NDArray:
    return (
        clip.reshape(clip.shape[0], 8, 4, 8, 4, 3)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(-1, 48)
    )


def list_patches(frames_dir: Path, max_clips: int | None = None) -> NDArray:
    all_patches: list[NDArray] = []
    npy_files = sorted(frames_dir.glob("*.npy"))
    if max_clips is not None:
        npy_files = npy_files[:max_clips]
    print(f"loading {len(npy_files)} clips from {frames_dir}")
    for i, fn in enumerate(npy_files):
        clip = np.load(fn)
        all_patches.append(_extract_patches(clip))
        if (i + 1) % 100 == 0 or i + 1 == len(npy_files):
            print(f"  {i + 1}/{len(npy_files)} clips loaded")
    return np.concatenate(all_patches, axis=0)


def extract_codebook(patches: NDArray, iters=20, chunk=8192):
    rng = np.random.default_rng(100)
    n = patches.shape[0]
    k = VOCAB_SIZE
    patches = patches.astype(np.float32) / 255.0
    centroids = patches[rng.choice(n, k, replace=False)]

    print(f"building codebook: k={k}, patches={n}, iters={iters}")
    for i in range(iters):
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
        print(f"  iter {i + 1}/{iters}: {int(empty.sum())} empty clusters")
    return centroids


def save_codebook(centroids: NDArray) -> None:
    np.save(CODEBOOK_FILE, centroids)


def load_codebook() -> NDArray:
    return np.load(CODEBOOK_FILE)


def get_codebook(
    patches: NDArray,
    iters: int = 20,
    chunk: int = 8192,
) -> NDArray:
    centroids = extract_codebook(patches, iters, chunk)
    np.save(CODEBOOK_FILE, centroids)
    print(
        f"saved codebook to {CODEBOOK_FILE} "
        f"({centroids.shape[0]}x{centroids.shape[1]}, {centroids.dtype})"
    )
    return centroids


def decode(ids: NDArray, centroids: NDArray) -> NDArray:
    return centroids[ids]


def encode(clip: NDArray, centroids: NDArray, chunk: int = 8192) -> NDArray:
    patches = _extract_patches(clip).astype(np.float32) / 255.0
    n = patches.shape[0]
    ids = np.empty(n, dtype=np.int16)
    for s in range(0, n, chunk):
        p = patches[s : s + chunk]
        d = ((p[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        ids[s : s + chunk] = d.argmin(1)
    return ids


def tokenize(frame_dir: Path, rebuild: bool = False) -> NDArray:
    if Path(CODEBOOK_FILE).exists() and not rebuild:
        print(f"loading codebook from {CODEBOOK_FILE}")
        return np.load(CODEBOOK_FILE)
    print(f"codebook {CODEBOOK_FILE} not found - building")
    patches = list_patches(frame_dir, 50)
    return get_codebook(patches)


def vocab_size() -> int:
    return VOCAB_SIZE


if __name__ == "__main__":
    codebook = tokenize(Path("frames"))
    print(f"{codebook.shape=} {codebook.dtype=}")
