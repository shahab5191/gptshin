"""Spiral toy dataset, ported 1:1 from stage1's ``toy.rs`` + ``rng.rs``.

Reproduces the exact same points (and labels) as the Rust version for a given
seed, so you can compare training curves across languages.
"""

from __future__ import annotations

import math

import numpy as np

MASK = (1 << 64) - 1


class Rng:
    """xorshift64* PRNG, mirroring ``stage1/src/rng.rs``."""

    def __init__(self, seed: int) -> None:
        # Rust: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
        self.state = ((seed * 0x9E37_79B9_7F4A_7C15) & MASK) | 1

    def next_u64(self) -> int:
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & MASK
        x ^= x >> 27
        self.state = x & MASK
        return (x * 0x2545_F491_4F6C_DD1D) & MASK

    def uniform(self) -> float:
        # Rust: (next_u64() >> 40) as F / (1u64 << 24) as F
        return (self.next_u64() >> 40) / (1 << 24)

    def normal(self) -> float:
        # Rust: Box-Muller
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_spiral(
    n: int = 100, k: int = 3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """k interleaved spiral arms with n points each.

    Returns ``(X, y)`` where ``X`` has shape ``(n * k, 2)`` and ``y`` has shape
    ``(n * k,)`` with integer labels ``0..k-1``.
    """
    rng = Rng(seed)
    x = np.zeros((n * k, 2), dtype=np.float32)
    y = np.zeros(n * k, dtype=np.int64)

    for c in range(k):
        for i in range(n):
            idx = c * n + i
            f = i / (n - 1)
            t = c * 4.0 + f * 4.0 + rng.normal() * 0.2
            x[idx, 0] = f * math.sin(t)
            x[idx, 1] = f * math.cos(t)
            y[idx] = c

    return x, y


if __name__ == "__main__":
    x, y = make_spiral(100, 3, seed=0)
    print("shape:", x.shape, y.shape)
    print("first 5 rows:")
    for i in range(5):
        print(f"  [{x[i, 0]:.17g}, {x[i, 1]:.17g}] -> {y[i]}")
    print("sum x:", repr(float(x.sum())))
    print("sum y:", repr(int(y.sum())))
