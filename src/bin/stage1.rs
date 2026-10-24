// src/bin/stage1.rs
use gptshin::ops::*;
use gptshin::rng::Rng;
use gptshin::toy::make_spiral;

const NAMES: [&str; 4] = ["w1", "b1", "w2", "b2"];

// h is the number of hidden units
struct Net {
    h: usize,
    w1: Vec<F>, // 2 × h
    b1: Vec<F>,
    w2: Vec<F>, // h × 3
    b2: Vec<F>,
}

// Cache holds intermediate results for the forward pass
struct Cache {
    z1: Vec<F>,
    a: Vec<F>,
    dlogits: Vec<F>,
}

impl Net {
    fn new(h: usize, rng: &mut Rng) -> Net {
        // He initialization: std = sqrt(2 / fan_in), explained in Stage 2
        let mut he = |fan_in: usize, len: usize| -> Vec<F> {
            let s = (2.0 / fan_in as F).sqrt();
            (0..len).map(|_| rng.normal() * s).collect()
        };
        let w1 = he(2, 2 * h);
        let w2 = he(h, h * 3);
        Net {
            h,
            w1,
            b1: vec![0.0; h],
            w2,
            b2: vec![0.0; 3],
        }
    }

    // x is a vector of input features
    // y is a vector of target labels
    // n is the number of examples
    fn forward(&self, x: &[F], n: usize, y: &Vec<usize>) -> Cache {
        assert_eq!(x.len(), 2 * n);
        assert_eq!(self.w1.len(), 2 * self.h);
        assert_eq!(self.w2.len(), 3 * self.h);
        assert_eq!(self.b1.len(), self.h);
        assert_eq!(self.b2.len(), 3);

        let mut z1 = vec![0.0; n * self.h];
        mm(x, &self.w1, &mut z1, n, 2, self.h);
        add_bias(&mut z1, &self.b1);
        let a = relu(&mut z1);

        let mut logits = vec![0.0; n * 3];
        mm(&a, &self.w2, &mut logits, n, self.h, 3);
        add_bias(&mut logits, &self.b2);
        let (loss, dlogits) = softmax_xent(&logits, y.as_slice(), n, 3);
        println!("{loss:.4} ");
        Cache { z1, a, dlogits }
    }

    fn backward(
        &self,
        cache: &Cache,
        n: usize,
        input: &[F],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut dw2 = vec![0.0; self.w2.len()];
        let mut db2 = vec![0.0; 3];
        mm_tn_acc(&cache.a, &cache.dlogits, &mut dw2, n, self.h, 3);
        for i in 0..n {
            for j in 0..3 {
                db2[j] += cache.dlogits[i * 3 + j];
            }
        }
        let mut da1 = vec![0.0; cache.a.len()];
        mm_nt_acc(&cache.dlogits, &self.w2, &mut da1, n, 3, self.h);
        let dz1: Vec<F> = da1
            .iter()
            .zip(cache.z1.iter())
            .map(|(da, z)| if *z > 0.0 { *da } else { 0.0 })
            .collect();
        let mut dw1 = vec![0.0; self.w1.len()];
        mm_tn_acc(&input, &dz1, &mut dw1, n, 2, self.h);
        let mut db1 = vec![0.0; self.b1.len()];
        for i in 0..n {
            for j in 0..self.b1.len() {
                db1[j] += dz1[i * self.h + j];
            }
        }
        (dw1, db1, dw2, db2)
    }
}

fn main() {
    let mut rng = Rng::new(0);
    let (x, y) = make_spiral(100, 3, &mut rng);
    let n = y.len();
    let net = Net::new(64, &mut rng);

    let cache = net.forward(&x, n, &y);
    let (dw1, db1, dw2, db2) = net.backward(&cache, n, &x);
    println!("dw1: {:?}", dw1);
    println!("db1: {:?}", db1);
    println!("dw2: {:?}", dw2);
    println!("db2: {:?}", db2);
}
