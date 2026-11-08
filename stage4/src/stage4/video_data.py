import os
import subprocess

import numpy as np

VIDEO_DIR = "videos"
FRAMES_DIR = "frames"
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def _cache_path(
    video_path: str, size: tuple[int, int], grayscale: bool, fps: float | None
) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    w, h = size
    mode = "gray" if grayscale else "rgb"
    suffix = f"_fps{fps:g}" if fps else ""
    return os.path.join(FRAMES_DIR, f"{stem}_{w}x{h}_{mode}{suffix}.npy")


def _extract(
    video_path: str, size: tuple[int, int], grayscale: bool, fps: float | None
) -> np.ndarray:
    w, h = size
    channels = 1 if grayscale else 3
    pix_fmt = "gray" if grayscale else "rgb24"
    # scale so the short side fits, then center-crop to exactly w x h (no stretching)
    filters = [
        f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=area",
        f"crop={w}:{h}",
    ]
    if fps:
        filters.insert(0, f"fps={fps}")
    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path, "-an",
        "-vf", ",".join(filters),
        "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
    ]  # fmt: skip
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    return frames.reshape(-1, h, w, channels)


def load_video(
    video_path: str,
    size: tuple[int, int] = (32, 32),
    grayscale: bool = False,
    fps: float | None = None,
) -> np.ndarray:
    cache = _cache_path(video_path, size, grayscale, fps)
    if os.path.exists(cache):
        return np.load(cache)

    print(f"extracting {video_path} -> {cache}")
    frames = _extract(video_path, size, grayscale, fps)
    os.makedirs(FRAMES_DIR, exist_ok=True)
    tmp = cache + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, frames)
    os.replace(tmp, cache)
    return frames


def load_videos(
    video_dir: str = VIDEO_DIR,
    size: tuple[int, int] = (32, 32),
    grayscale: bool = False,
    fps: float | None = None,
) -> list[np.ndarray]:
    names = sorted(n for n in os.listdir(video_dir) if n.lower().endswith(VIDEO_EXTS))
    if not names:
        raise FileNotFoundError(f"no videos found in {video_dir}/")
    return [
        load_video(os.path.join(video_dir, n), size, grayscale, fps) for n in names
    ]


if __name__ == "__main__":
    for v in load_videos():
        print(v.shape, v.dtype)
