from pathlib import Path

from numpy.typing import NDArray
from PIL import Image


def save_gif(frames: NDArray, path: Path, duration=200):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
