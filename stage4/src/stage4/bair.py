"""BAIR robot pushing dataset: a robot arm pushing objects on a table.

43,264 train + 256 test clips, 30 frames each, 64x64 colour, fixed camera.
The official download is one ~32 GB tar of TFRecord files:

    http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar

convert() streams that tar once and writes every clip's frames to a raw uint8
file per split (frames/bair/{train,test}.u8). tokens() turns those into patch
ids with a codebook and caches the result, so later runs start in seconds.

TFRecords normally need TensorFlow to read. The format is simple enough to
parse by hand: each record is [length][crc][protobuf bytes][crc], and the
protobuf is a tf.train.Example, a map from feature name to a list of bytes.
"""

import struct
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from stage4 import tokenizer as tok

TAR_FILE = Path("videos") / "bair" / "bair_robot_pushing_dataset_v0.tar"
OUT_DIR = Path("frames") / "bair"
CLIP_FRAMES = 30
SIZE = 64
FRAME_BYTES = SIZE * SIZE * 3
IMAGE_KEY = "{t}/image_aux1/encoded"  # the camera view the BAIR papers use


# --- reading TFRecords without TensorFlow ---------------------------------


def _varint(buf: memoryview, i: int) -> tuple[int, int]:
    """Protobuf variable-length int: 7 bits per byte, high bit = more bytes."""
    result = shift = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if b < 0x80:
            return result, i
        shift += 7


def _fields(buf: memoryview) -> Iterator[tuple[int, memoryview]]:
    """Yields (field number, bytes) for every length-delimited field of a message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 2:  # length-delimited: strings, bytes, sub-messages
            n, i = _varint(buf, i)
            yield field, buf[i : i + n]
            i += n
        elif wire == 0:  # varint
            _, i = _varint(buf, i)
        elif wire == 1:  # 64-bit
            i += 8
        elif wire == 5:  # 32-bit
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")


def _features(example: memoryview) -> dict[str, memoryview]:
    """tf.train.Example -> {feature name: first bytes value}.

    Example{1: Features}, Features{1: repeated map entry {1: key, 2: Feature}},
    Feature{1: BytesList}, BytesList{1: repeated bytes}. Float/int features
    (actions, arm position) are skipped; we only need the images.
    """
    out: dict[str, memoryview] = {}
    for _, features in _fields(example):
        for _, entry in _fields(features):
            key = value = None
            for f, v in _fields(entry):
                if f == 1:
                    key = bytes(v).decode()
                elif f == 2:
                    value = v
            if key is None or value is None:
                continue
            for kind, lst in _fields(value):
                if kind == 1:  # bytes_list
                    for _, b in _fields(lst):
                        out[key] = b
                        break
    return out


def _records(f) -> Iterator[memoryview]:
    """TFRecord file -> raw record payloads (CRCs are skipped)."""
    while header := f.read(12):
        (n,) = struct.unpack("<Q", header[:8])
        data = f.read(n)
        f.read(4)
        yield memoryview(data)


def _clip(record: memoryview) -> NDArray:
    feats = _features(record)
    frames = [
        np.frombuffer(feats[IMAGE_KEY.format(t=t)], dtype=np.uint8)
        for t in range(CLIP_FRAMES)
    ]
    return np.stack(frames).reshape(CLIP_FRAMES, SIZE, SIZE, 3)


# --- conversion and caching -----------------------------------------------


def _frames_path(split: str) -> Path:
    return OUT_DIR / f"{split}.u8"


def convert(tar_path: Path = TAR_FILE) -> None:
    """Stream the tar once and append every clip to frames/bair/{split}.u8."""
    if all(_frames_path(s).exists() for s in ("train", "test")):
        return
    if not tar_path.exists():
        raise SystemExit(f"{tar_path} not found - download it from the URL in bair.py")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = {s: _frames_path(s).with_suffix(".partial") for s in ("train", "test")}
    counts = {"train": 0, "test": 0}
    print(f"converting {tar_path} (one pass over ~32 GB)")
    with (
        open(tmp["train"], "wb") as train_out,
        open(tmp["test"], "wb") as test_out,
        tarfile.open(tar_path, mode="r|") as tar,  # "r|" = stream, no seeking
    ):
        outs = {"train": train_out, "test": test_out}
        for member in tar:
            if not member.name.endswith(".tfrecords"):
                continue
            split = "test" if "/test/" in member.name else "train"
            f = tar.extractfile(member)
            assert f is not None
            for record in _records(f):
                outs[split].write(_clip(record).tobytes())
                counts[split] += 1
            print(f"  {member.name}: {counts['train']} train, {counts['test']} test clips")
    for s, path in tmp.items():  # only now: a crash above leaves .partial files
        path.rename(_frames_path(s))
    print(f"done: {counts}")


def load_frames(split: str) -> NDArray:
    """(N, 30, 64, 64, 3) uint8, memory-mapped: read from disk only when touched."""
    path = _frames_path(split)
    n = path.stat().st_size // (CLIP_FRAMES * FRAME_BYTES)
    return np.memmap(path, dtype=np.uint8, mode="r", shape=(n, CLIP_FRAMES, SIZE, SIZE, 3))


def codebook(path: Path, k: int, n_frames: int = 4000) -> NDArray:
    """k-means codebook from the patches of n_frames random train frames
    (4000 frames * 256 patches = ~1M patches, like the KTH codebook)."""
    if path.exists():
        print(f"loading codebook from {path}")
        return np.load(path)
    frames = load_frames("train")
    rng = np.random.default_rng(0)
    clips = rng.integers(0, len(frames), n_frames)
    ts = rng.integers(0, CLIP_FRAMES, n_frames)
    picked = np.stack([frames[c, t] for c, t in zip(clips, ts)])
    return tok.get_codebook(tok._extract_patches(picked), path=path, k=k)


@torch.no_grad()
def tokens(split: str, centroids: NDArray, codebook_path: Path) -> NDArray:
    """(N, 30 * 256) int16 patch ids for every clip of a split, cached per codebook.

    Same nearest-centroid search as tokenizer.encode, but on the GPU: 43k clips
    are 334M patches, which takes numpy on the CPU a long while.
    """
    path = OUT_DIR / f"{split}_{codebook_path.stem}.npy"
    if path.exists():
        return np.load(path)
    frames = load_frames(split)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c = torch.from_numpy(centroids).to(device)
    c_sq = (c**2).sum(1)
    tpf = tok.tokens_per_frame(SIZE, SIZE)
    out = np.empty((len(frames), CLIP_FRAMES * tpf), dtype=np.int16)
    batch = 64  # clips per GPU batch: 64 * 30 * 256 patches
    print(f"encoding {len(frames)} {split} clips with {path.stem}")
    for s in range(0, len(frames), batch):
        clips = np.asarray(frames[s : s + batch])
        p = torch.from_numpy(tok._extract_patches(clips.reshape(-1, SIZE, SIZE, 3)))
        p = p.to(device).float() / 255.0
        # the distance table is patches x codebook floats; 64K patches at a time
        # keeps it at 1 GB even with 4096 codebook entries
        ids = torch.cat(
            [
                (c_sq[None, :] - 2.0 * (q @ c.T)).argmin(1)  # ||c||^2 - 2 p.c
                for q in p.split(65536)
            ]
        )
        out[s : s + batch] = ids.view(len(clips), -1).cpu().numpy()
        if (s // batch) % 100 == 0:
            print(f"  {s + len(clips)}/{len(frames)}")
    np.save(path, out)
    return out
