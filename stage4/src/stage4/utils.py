import os
import select
import sys
import termios
import tty
from pathlib import Path

from numpy.typing import NDArray
from PIL import Image


def save_gif(frames: NDArray, path: Path | str, duration=200):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)


class KeyListener:
    """Non-blocking single-key reader for the terminal running the script.

    Inside the `with` block the terminal is in cbreak mode: keys arrive without
    Enter and aren't echoed; Ctrl+C still works. The terminal is restored on
    exit. If stdin isn't a terminal (piped, nohup, IDE), it never fires.
    """

    def __enter__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        if self.fd is not None:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def pressed(self, key: str) -> bool:
        """True if `key` was typed since the last call (other keys are dropped)."""
        if self.fd is None:
            return False
        hit = False
        # os.read on the raw fd: sys.stdin.read would buffer extra keys out of
        # select()'s sight, so a "q" typed after another key could be missed
        while select.select([self.fd], [], [], 0)[0]:
            if os.read(self.fd, 1) == key.encode():
                hit = True
        return hit
