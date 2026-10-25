// src/net.rs
use crate::ops::*;
use crate::rng::Rng;

// h is the number of hidden units
pub struct Net {
    pub h: usize,
    pub lr: F, // learning rate
    pub ws: Vec<Vec<F>>,
    pub bs: Vec<Vec<F>>,
    pub dimensions: Vec<(usize, usize)>,
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
        let dimensions = vec![(2, h), (h, h), (h, h / 2), (h / 2, 3)];
        let ws = vec![he(2, 2 * h), he(h, h * h), he(h, h * (h / 2)), he(h / 2, (h / 2) * 3)];
        let bs = vec![vec![0.0; h], vec![0.0; h], vec![0.0; h / 2], vec![0.0; 3]];
        Net {
            h,
            lr: 0.1,
            ws,
            bs,
            dimensions
        }
    }

    // x is a vector of input features
    // y is a vector of target labels
    // n is the number of examples
    pub fn forward(&self, x: &[F], n: usize, y: &Vec<usize>) -> (Cache, F) {
        assert_eq!(x.len(), self.dimensions[0].0 * n);

        let layers_count: usize = self.ws.len();

        let mut z: Vec<Vec<F>> = Vec::new();
        let mut a: Vec<Vec<F>> = Vec::new();
        let mut act = x.to_vec();
        for i in 0..layers_count - 1 {
            let (in_dim, out_dim) = self.dimensions[i];
            let mut current_z = vec![0.0; n * out_dim];
            mm(&act, &self.ws[i], &mut current_z, n, in_dim, out_dim);
            add_bias(&mut current_z, &self.bs[i]);
            let act_i = relu(&current_z);
            z.push(current_z);
            a.push(act_i.clone());
            act = act_i;
        }

        let last = layers_count - 1;
        let (in_dim, out_dim) = self.dimensions[last];
        let mut logits = vec![0.0; n * out_dim];
        mm(&a.last().unwrap(), &self.ws[last], &mut logits, n, in_dim, out_dim);
        add_bias(&mut logits, &self.bs[last]);
        let (loss, dlogits) = softmax_xent(&logits, y.as_slice(), n, out_dim);
        (Cache { z, a, dlogits }, loss)
    }

    pub fn backward(
        &self,
        cache: &Cache,
        n: usize,
        input: &[F],
    ) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
        let layers_count: usize = self.ws.len();
        let mut dws: Vec<Vec<F>> = vec![Vec::new(); layers_count];
        let mut dbs: Vec<Vec<F>> = vec![Vec::new(); layers_count];

        // output layer
        let (in_dim, out_dim) = self.dimensions[layers_count - 1];
        let mut dw = vec![0.0; self.ws[layers_count - 1].len()];
        let mut db = vec![0.0; out_dim];
        mm_tn_acc(cache.a.last().unwrap(), &cache.dlogits, &mut dw, n, in_dim, out_dim);
        for i in 0..n {
            for j in 0..out_dim {
                db[j] += cache.dlogits[i * out_dim + j];
            }
        }
        dws[layers_count - 1] = dw;
        dbs[layers_count - 1] = db;

        // hidden layers, in reverse
        let mut d_up = cache.dlogits.clone();
        for i in (0..layers_count - 1).rev() {
            let (in_dim, out_dim) = self.dimensions[i];
            let next_out = self.dimensions[i + 1].1;
            // da = d_up · ws[i+1]ᵀ, with d_up (n × next_out) and ws[i+1] (out_dim × next_out)
            let mut da = vec![0.0; n * out_dim];
            mm_nt_acc(&d_up, &self.ws[i + 1], &mut da, n, next_out, out_dim);
            let dz: Vec<F> = da
                .iter()
                .zip(cache.z[i].iter())
                .map(|(da, z)| if *z > 0.0 { *da } else { 0.0 })
                .collect();
            let layer_input = if i == 0 { input } else { &cache.a[i - 1] };
            let mut dw = vec![0.0; self.ws[i].len()];
            mm_tn_acc(layer_input, &dz, &mut dw, n, in_dim, out_dim);
            let mut db = vec![0.0; out_dim];
            for j in 0..n {
                for k in 0..out_dim {
                    db[k] += dz[j * out_dim + k];
                }
            }
            dws[i] = dw;
            dbs[i] = db;
            d_up = dz;
        }
        (dws, dbs)
    }

    pub fn learn(&mut self, dws: Vec<Vec<F>>, dbs: Vec<Vec<F>>) {
        assert_eq!(self.ws.len(), dws.len());
        assert_eq!(self.ws.len(), dbs.len());
        for i in 0..self.ws.len() {
            weighted_sub(&mut self.ws[i], &dws[i], self.lr);
            weighted_sub(&mut self.bs[i], &dbs[i], self.lr);
        }
    }
}
