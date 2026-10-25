// src/bin/stage1.rs
use gptshin::net::Net;
use gptshin::rng::Rng;
use gptshin::toy::make_spiral;

fn main() {
    let mut rng = Rng::new(0);
    let (x, y) = make_spiral(100, 3, &mut rng);
    let n = y.len();
    let mut net = Net::new(64, &mut rng);

    let mut last_loss = 1.0;
    while last_loss >= 0.01 {
        let (cache, loss) = net.forward(&x, n, &y);
        last_loss = loss;
        println!("loss: {}", loss);
        let (dws, dbs) = net.backward(&cache, n, &x);
        net.learn(dws, dbs);
    }
}
