use std::time::Instant;

use gptshin::ops::*;
use gptshin::rng::Rng;

fn time(name: &str, reps: u32, mut f: impl FnMut()) {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    let dt = t.elapsed() / reps;
    println!("    {name:<26} {dt:>10.2?}");
}

fn bench_size(n: usize, h: usize, reps: u32) {
    let mut rng = Rng::new(7);
    let mut rnd = |len: usize| -> Vec<F> { (0..len).map(|_| rng.normal()).collect() };

    let x = rnd(n * 2);
    let a = rnd(n * h);
    let w1 = rnd(2 * h);
    let w2 = rnd(h * 3);
    let dl = rnd(n * 3);
    let dz = rnd(n * h);
    let logits = rnd(n * 3);
    let y: Vec<usize> = (0..n).map(|i| i % 3).collect();

    let mut z1 = vec![0.0; n * h];
    let mut logits_out = vec![0.0; n * 3];
    let mut gw1 = vec![0.0; 2 * h];
    let mut gw2 = vec![0.0; h * 3];
    let mut dz_out = vec![0.0; n * h];

    time("mm w1  (n×2 · 2×h)", reps, || mm(&x, &w1, &mut z1, n, 2, h));
    time("mm w2  (n×h · h×3)", reps, || {
        mm(&a, &w2, &mut logits_out, n, h, 3)
    });
    time("mm_tn dW2 (h×3)", reps, || {
        mm_tn_acc(&a, &dl, &mut gw2, n, h, 3)
    });
    time("mm_tn dW1 (2×h)", reps, || {
        mm_tn_acc(&x, &dz, &mut gw1, n, 2, h)
    });
    time("mm_nt dz  (n×h)", reps, || {
        mm_nt_acc(&dl, &w2, &mut dz_out, n, 3, h)
    });
    time("softmax_xent (n×3)", reps, || {
        softmax_xent(&logits, &y, n, 3);
    });
    time("accuracy (n×3)", reps, || {
        accuracy(&logits, &y, 3);
    });
}

fn main() {
    let cpus = rayon::current_num_threads();
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(cpus)
        .build()
        .unwrap();

    println!("CPUs available: {cpus}");

    let cases: [(&str, usize, usize, u32); 2] =
        [("small (model size)", 300, 64, 2000), ("large", 20_000, 256, 20)];

    for (label, n, h, reps) in cases {
        println!("\n--- {label}: n={n} h={h} ---");
        println!("  serial (1 thread):");
        serial.install(|| bench_size(n, h, reps));
        println!("  parallel ({cpus} threads):");
        parallel.install(|| bench_size(n, h, reps));
    }
}
