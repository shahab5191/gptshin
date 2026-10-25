// src/bin/stage1.rs
use gptshin::net::Net;
use gptshin::ops::*;
use gptshin::rng::Rng;
use gptshin::toy::make_spiral;

fn main() {
    let mut rng = Rng::new(0);
    let (x, y) = make_spiral(100, 3, &mut rng);
    let n = y.len();
    let mut net = Net::new(64, &mut rng);

    let mut delta_loss: F = 1.0;
    let mut last_loss = 1.0;
    while last_loss >= 0.01 {
        let (cache, loss) = net.forward(&x, n, &y);
        delta_loss = F::abs(last_loss - loss);
        last_loss = loss;
        println!("loss: {}", loss);
        println!("delta_loss: {}", delta_loss);
        let (dw1, db1, dw2, db2) = net.backward(&cache, n, &x);
        net.learn(dw1, db1, dw2, db2);
    }
}
