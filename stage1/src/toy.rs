// src/toy.rs
use crate::ops::F;
use crate::rng::Rng;

/// k interleaved spiral arms with n points each. Returns (points as n*k rows of 2, labels).
pub fn make_spiral(n: usize, k: usize, rng: &mut Rng) -> (Vec<F>, Vec<usize>) {
    let mut x = vec![0.0; n * k * 2];
    let mut y = vec![0usize; n * k];
    for c in 0..k {
        for i in 0..n {
            let idx = c * n + i;
            let f = (i as F) / ((n - 1) as F);
            let t = (c as F) * 4.0 + f * 4.0 + rng.normal() * 0.2;
            x[idx * 2] = f * t.sin();
            x[idx * 2 + 1] = f * t.cos();
            y[idx] = c;
        }
    }
    (x, y)
}
