// src/ops.rs
use rayon::prelude::*;

pub type F = f64;

/// C (m×n) = A (m×k) · B (k×n)
pub fn mm(a: &[F], b: &[F], c: &mut [F], m: usize, k: usize, n: usize) {
    debug_assert_eq!(c.len(), m * n);
    c.fill(0.0);
    c.par_chunks_mut(n).enumerate().for_each(|(i, crow)| {
        for p in 0..k {
            for j in 0..n {
                crow[j] += a[i * k + p] * b[p * n + j];
            }
        }
    });
}

/// C (k×n) += Aᵀ · B, with A (m×k) and B (m×n). Used for weight gradients.
pub fn mm_tn_acc(a: &[F], b: &[F], c: &mut [F], m: usize, k: usize, n: usize) {
    c.par_chunks_mut(n).enumerate().for_each(|(p, crow)| {
        for i in 0..m {
            let aip = a[i * k + p];
            let brow = &b[i * n..(i + 1) * n];
            for j in 0..n {
                crow[j] += aip * brow[j];
            }
        }
    });
}

/// C (m×k) += A · Bᵀ, with A (m×n) and B (k×n). Used for input gradients.
pub fn mm_nt_acc(a: &[F], b: &[F], c: &mut [F], m: usize, n: usize, k: usize) {
    debug_assert_eq!(c.len(), m * k);
    c.par_chunks_mut(k).enumerate().for_each(|(i, crow)| {
        let arow = &a[i * n..(i + 1) * n];
        for p in 0..k {
            let brow = &b[p * n..(p + 1) * n];
            let mut s = 0.0;
            for j in 0..n {
                s += arow[j] * brow[j];
            }
            crow[p] += s;
        }
    });
}

/// Mean cross-entropy over n rows of v logits, and its gradient w.r.t. the logits.
pub fn softmax_xent(logits: &[F], y: &[usize], n: usize, v: usize) -> (F, Vec<F>) {
    let mut d = vec![0.0; n * v];
    let loss = d
        .par_chunks_mut(v)
        .enumerate()
        .map(|(i, drow)| {
            let row = &logits[i * v..(i + 1) * v];
            let m = row.iter().cloned().fold(F::NEG_INFINITY, F::max); // stability
            let mut sum = 0.0;
            for j in 0..v {
                let e = (row[j] - m).exp();
                drow[j] = e;
                sum += e;
            }
            for j in 0..v {
                drow[j] /= sum;
            }
            let l = drow[y[i]].ln();
            drow[y[i]] -= 1.0; // p - onehot
            -l
        })
        .sum::<F>();
    let nf = n as F;
    d.par_iter_mut().for_each(|x| *x /= nf);
    (loss / nf, d)
}

pub fn accuracy(logits: &[F], y: &[usize], v: usize) -> F {
    let ok = y
        .par_iter()
        .enumerate()
        .map(|(i, &target)| {
            let row = &logits[i * v..(i + 1) * v];
            let mut best = 0;
            for j in 1..v {
                if row[j] > row[best] {
                    best = j;
                }
            }
            (best == target) as usize
        })
        .sum::<usize>();
    ok as F / y.len() as F
}

// b is one row of biases, a is a matrix of n rows
pub fn add_bias(a: &mut [F], b: &[F]) {
    assert!(a.len() % b.len() == 0);
    a.par_chunks_mut(b.len()).for_each(|row| {
        row.iter_mut().zip(b).for_each(|(a, b)| *a += *b);
    });
}

pub fn relu(a: &[F]) -> Vec<F> {
    a.par_iter().map(|x| x.max(0.0)).collect()
}
