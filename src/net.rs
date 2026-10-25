// src/net.rs
use crate::ops::*;
use crate::rng::Rng;

// h is the number of hidden units
pub struct Net {
    pub h: usize,
    pub lr: F, // learning rate
    pub ws: Vec<Vec<F>>,
    pub bs: Vec<Vec<F>>,
}

// Cache holds intermediate results for the forward pass
pub struct Cache {
    pub z: Vec<Vec<F>>,
    pub a: Vec<Vec<F>>,
    pub dlogits: Vec<F>,
}

impl Net {
    pub fn new(h: usize, rng: &mut Rng) -> Net {
        // He initialization: std = sqrt(2 / fan_in), explained in Stage 2
        let mut he = |fan_in: usize, len: usize| -> Vec<F> {
            let s = (2.0 / fan_in as F).sqrt();
            (0..len).map(|_| rng.normal() * s).collect()
        };
        let ws = vec![he(2, 2 * h), he(h, h * h), he(h, h * 3)];
        let bs = vec![vec![0.0; h], vec![0.0; h], vec![0.0; 3]];
        Net {
            h,
            lr: 0.1,
            ws,
            bs,
        }
    }

    // x is a vector of input features
    // y is a vector of target labels
    // n is the number of examples
    pub fn forward(&self, x: &[F], n: usize, y: &Vec<usize>) -> (Cache, F) {
        assert_eq!(x.len(), 2 * n);

        let layers_count: usize = self.ws.len();

        let mut z: Vec<Vec<F>> = Vec::new();
        let mut a: Vec<Vec<F>> = Vec::new();
        let mut act = x.to_vec();
        for i in 0..layers_count - 1 {
            let mut current_z = vec![0.0; n * self.bs[i].len()];
            mm(&act, &self.ws[i], &mut current_z, n, act.len() / n, self.bs[i].len());
            add_bias(&mut current_z, &self.bs[i]);
            let act_i = relu(&current_z);
            z.push(current_z);
            a.push(act_i.clone());
            act = act_i;
        }

        let mut logits = vec![0.0; n * 3];
        mm(&a.last().unwrap(), &self.ws[layers_count - 1], &mut logits, n, self.h, 3);
        add_bias(&mut logits, &self.bs[layers_count - 1]);
        let (loss, dlogits) = softmax_xent(&logits, y.as_slice(), n, 3);
        (Cache { z, a, dlogits }, loss)
    }

    pub fn backward(
        &self,
        cache: &Cache,
        n: usize,
        input: &[F],
    ) -> (Vec<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>, Vec<F>) {
        let mut dw3 = vec![0.0; self.ws[2].len()];
        let mut db3 = vec![0.0; 3];
        mm_tn_acc(cache.a.last().unwrap(), &cache.dlogits, &mut dw3, n, self.h, 3);
        for i in 0..n {
            for j in 0..3 {
                db3[j] += cache.dlogits[i * 3 + j];
            }
        }


        let mut da2 = vec![0.0; cache.a[1].len()];
        mm_nt_acc(&cache.dlogits, &self.ws[2], &mut da2, n, 3, self.h);
        let dz2: Vec<F> = da2
            .iter()
            .zip(cache.z[1].iter())
            .map(|(da, z)| if *z > 0.0 { *da } else { 0.0 })
            .collect();
        let mut dw2 = vec![0.0; self.ws[1].len()];
        mm_tn_acc(&cache.a[0], &dz2, &mut dw2, n, self.h, self.h);
        let mut db2 = vec![0.0; self.bs[1].len()];
        for i in 0..n {
            for j in 0..self.bs[1].len() {
                db2[j] += dz2[i * self.h + j];
            }
        }

        let mut da1 = vec![0.0; cache.a[0].len()];
        mm_nt_acc(&dz2, &self.ws[1], &mut da1, n, self.h, self.h);
        let dz1: Vec<F> = da1
            .iter()
            .zip(cache.z[0].iter())
            .map(|(da, z)| if *z > 0.0 { *da } else { 0.0 })
            .collect();
        let mut dw1 = vec![0.0; self.ws[0].len()];
        mm_tn_acc(input, &dz1, &mut dw1, n, 2, self.h);
        let mut db1 = vec![0.0; self.bs[0].len()];
        for i in 0..n {
            for j in 0..self.bs[0].len() {
                db1[j] += dz1[i * self.h + j];
            }
        }
        (dw3, db3, dw2, db2, dw1, db1)
    }

    pub fn learn(&mut self, dw1: Vec<F>, db1: Vec<F>, dw2: Vec<F>, db2: Vec<F>, dw3: Vec<F>, db3: Vec<F>) {
        weighted_sub(&mut self.ws[0], &dw1, self.lr);
        weighted_sub(&mut self.bs[0], &db1, self.lr);
        weighted_sub(&mut self.ws[1], &dw2, self.lr);
        weighted_sub(&mut self.bs[1], &db2, self.lr);
        weighted_sub(&mut self.ws[2], &dw3, self.lr);
        weighted_sub(&mut self.bs[2], &db3, self.lr);
    }
}
