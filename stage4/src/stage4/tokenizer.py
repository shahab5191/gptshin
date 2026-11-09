from pathlib import Path

import numpy as np
from numpy._typing import NDArray

CODEBOOK_FILE = "codebook.npy"
VOCAB_SIZE = 512
PATCH = 4  # patch side in pixels
PATCH_DIM = PATCH * PATCH * 3  # 48 values per RGB patch


def tokens_per_frame(h: int, w: int) -> int:
    return (h // PATCH) * (w // PATCH)


def _extract_patches(clip: NDArray) -> NDArray:
    # (T, H, W, 3) -> (T * gh * gw, 48), patches in row-major order within a frame
    t, h, w, c = clip.shape
    gh, gw = h // PATCH, w // PATCH
    return (
        clip.reshape(t, gh, PATCH, gw, PATCH, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(-1, PATCH_DIM)
    )


def decode_frames(ids, centroids: NDArray, h: int, w: int) -> NDArray:
    """Inverse of encode: token ids (T * gh * gw,) -> uint8 frames (T, H, W, 3)."""
    gh, gw = h // PATCH, w // PATCH
    patches = centroids[np.asarray(ids)]
    frames = (
        patches.reshape(-1, gh, gw, PATCH, PATCH, 3)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(-1, h, w, 3)
    )
    return (frames * 255).clip(0, 255).astype(np.uint8)


def _nearest(p: NDArray, centroids: NDArray) -> NDArray:
    # argmin_k ||p - c_k||^2 = argmin_k (||c_k||^2 - 2 p.c_k); ||p||^2 is constant per row
    d = (centroids**2).sum(1)[None, :] - 2.0 * (p @ centroids.T)
    return d.argmin(1)


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


def extract_codebook(patches: NDArray, iters=20, chunk=8192, k: int = VOCAB_SIZE):
    rng = np.random.default_rng(100)
    n = patches.shape[0]
    patches = patches.astype(np.float32) / 255.0
    centroids = patches[rng.choice(n, k, replace=False)]

    print(f"building codebook: k={k}, patches={n}, iters={iters}")
    for i in range(iters):
        labels = np.empty(n, dtype=np.int32)
        for s in range(0, n, chunk):
            labels[s : s + chunk] = _nearest(patches[s : s + chunk], centroids)
        sums = np.zeros((k, PATCH_DIM), dtype=np.float64)
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
    path: Path | str = CODEBOOK_FILE,
    k: int = VOCAB_SIZE,
) -> NDArray:
    centroids = extract_codebook(patches, iters, chunk, k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, centroids)
    print(
        f"saved codebook to {path} "
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
        ids[s : s + chunk] = _nearest(patches[s : s + chunk], centroids)
    return ids


def tokenize(
    frame_dir: Path,
    rebuild: bool = False,
    codebook_file: Path | str = CODEBOOK_FILE,
    max_clips: int | None = 50,
    max_patches: int = 1_000_000,
) -> NDArray:
    if Path(codebook_file).exists() and not rebuild:
        print(f"loading codebook from {codebook_file}")
        return np.load(codebook_file)
    print(f"codebook {codebook_file} not found - building")
    patches = list_patches(frame_dir, max_clips)
    if len(patches) > max_patches:  # random subset keeps k-means fast
        rng = np.random.default_rng(0)
        patches = patches[rng.choice(len(patches), max_patches, replace=False)]
    return get_codebook(patches, path=codebook_file)


def vocab_size() -> int:
    return VOCAB_SIZE


if __name__ == "__main__":
    codebook = tokenize(Path("frames"))
    print(f"{codebook.shape=} {codebook.dtype=}")
