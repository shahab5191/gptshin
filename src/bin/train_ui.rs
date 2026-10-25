// src/bin/train_ui.rs
// A visualization dashboard for the Net in src/net.rs.
use eframe::egui;
use egui_plot::{Heatmap, Legend, Line, MarkerShape, Plot, PlotPoint, PlotPoints, Points};
use rayon::prelude::*;
use gptshin::net::Net;
use gptshin::ops::*;
use gptshin::rng::Rng;
use gptshin::toy::make_spiral;

const GRID: usize = 60;

fn class_color(i: usize, alpha: u8) -> egui::Color32 {
    match i {
        0 => egui::Color32::from_rgba_unmultiplied(220, 50, 47, alpha),
        1 => egui::Color32::from_rgba_unmultiplied(0, 160, 90, alpha),
        _ => egui::Color32::from_rgba_unmultiplied(38, 139, 210, alpha),
    }
}

fn argmax(p: &[F; 3]) -> usize {
    let mut best = 0;
    for j in 1..3 {
        if p[j] > p[best] {
            best = j;
        }
    }
    best
}

fn compute_bounds(x: &[F]) -> [f64; 4] {
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for i in (0..x.len()).step_by(2) {
        xmin = xmin.min(x[i]);
        xmax = xmax.max(x[i]);
        ymin = ymin.min(x[i + 1]);
        ymax = ymax.max(x[i + 1]);
    }
    let px = (xmax - xmin) * 0.05;
    let py = (ymax - ymin) * 0.05;
    [xmin - px, xmax + px, ymin - py, ymax + py]
}

fn group_by_class(x: &[F], y: &[usize]) -> [Vec<[f64; 2]>; 3] {
    let mut out: [Vec<[f64; 2]>; 3] = Default::default();
    for i in 0..y.len() {
        out[y[i]].push([x[2 * i], x[2 * i + 1]]);
    }
    out
}

// Softmax probabilities for a single input point, without changing Net.
fn predict(net: &Net, px: F, py: F) -> [F; 3] {
    let n = 1;
    let layers = net.ws.len();
    let mut act = vec![px, py];
    for i in 0..layers {
        let (in_dim, out_dim) = net.dimensions[i];
        let mut next = vec![0.0; n * out_dim];
        mm(&act, &net.ws[i], &mut next, n, in_dim, out_dim);
        add_bias(&mut next, &net.bs[i]);
        act = if i == layers - 1 { next } else { relu(&next) };
    }

    let logits = act;
    let m = logits.iter().cloned().fold(F::NEG_INFINITY, F::max);
    let mut p = [0.0; 3];
    let mut sum = 0.0;
    for j in 0..3 {
        p[j] = (logits[j] - m).exp();
        sum += p[j];
    }
    for j in 0..3 {
        p[j] /= sum;
    }
    p
}

// Decision boundary: GRID×GRID grid, value = predicted class at each cell.
fn compute_boundary(net: &Net, bounds: [f64; 4]) -> Vec<f64> {
    let [xmin, xmax, ymin, ymax] = bounds;
    let mut values = vec![0.0; GRID * GRID];
    values
        .par_chunks_mut(GRID)
        .enumerate()
        .for_each(|(iy, row)| {
            let py = ymin + (ymax - ymin) * (iy as f64 + 0.5) / GRID as f64;
            for (ix, v) in row.iter_mut().enumerate() {
                let px = xmin + (xmax - xmin) * (ix as f64 + 0.5) / GRID as f64;
                *v = argmax(&predict(net, px, py)) as f64;
            }
        });
    values
}

struct App {
    net: Net,
    x: Vec<F>,
    y: Vec<usize>,
    n: usize,
    loss_hist: Vec<[f64; 2]>,
    acc_hist: Vec<[f64; 2]>,
    seed: u64,
    prev_loss: Option<F>,
    last_loss: F,
    last_delta: F,
    last_acc: F,
    train_steps: u32,
    training: bool,
    log_scale: bool,
    bounds: [f64; 4],
    data_by_class: [Vec<[f64; 2]>; 3],
    boundary: Vec<f64>,
    clicked: Option<[f64; 2]>,
    probs: Option<[F; 3]>,
}

impl App {
    fn new() -> App {
        let seed = 0;
        let mut rng = Rng::new(seed);
        let (x, y) = make_spiral(100, 3, &mut rng);
        let n = y.len();
        let net = Net::new(64, &mut rng);
        let bounds = compute_bounds(&x);
        let data_by_class = group_by_class(&x, &y);
        let boundary = compute_boundary(&net, bounds);
        App {
            net,
            x,
            y,
            n,
            loss_hist: Vec::new(),
            acc_hist: Vec::new(),
            seed,
            prev_loss: None,
            last_loss: f64::NAN,
            last_delta: f64::NAN,
            last_acc: f64::NAN,
            train_steps: 100,
            training: false,
            log_scale: false,
            bounds,
            data_by_class,
            boundary,
            clicked: None,
            probs: None,
        }
    }

    fn step_once(&mut self) {
        let (cache, loss) = self.net.forward(&self.x, self.n, &self.y);
        let delta = self.prev_loss.map(|p| (p - loss).abs()).unwrap_or(loss);
        self.prev_loss = Some(loss);

        // recompute logits for accuracy without changing Net
        let last = self.net.ws.len() - 1;
        let (in_dim, out_dim) = self.net.dimensions[last];
        let mut logits = vec![0.0; self.n * out_dim];
        let a_last = cache.a.last().unwrap();
        mm(a_last, &self.net.ws[last], &mut logits, self.n, in_dim, out_dim);
        add_bias(&mut logits, &self.net.bs[last]);
        let acc = accuracy(&logits, &self.y, out_dim);

        let (dws, dbs) = self.net.backward(&cache, self.n, &self.x);
        self.net.learn(dws, dbs);

        let step = self.loss_hist.len() as f64;
        self.loss_hist.push([step, loss]);
        self.acc_hist.push([step, acc]);
        self.last_loss = loss;
        self.last_delta = delta;
        self.last_acc = acc;
    }

    fn recompute_boundary(&mut self) {
        self.boundary = compute_boundary(&self.net, self.bounds);
    }

    fn run_steps(&mut self, steps: u32) {
        for _ in 0..steps {
            self.step_once();
        }
    }

    fn reset(&mut self) {
        self.seed += 1;
        let mut rng = Rng::new(self.seed);
        let (x, y) = make_spiral(100, 3, &mut rng);
        self.n = y.len();
        self.x = x;
        self.y = y;
        self.net = Net::new(64, &mut rng);
        self.bounds = compute_bounds(&self.x);
        self.data_by_class = group_by_class(&self.x, &self.y);
        self.boundary = compute_boundary(&self.net, self.bounds);
        self.loss_hist.clear();
        self.acc_hist.clear();
        self.prev_loss = None;
        self.last_loss = f64::NAN;
        self.last_delta = f64::NAN;
        self.last_acc = f64::NAN;
        self.clicked = None;
        self.probs = None;
    }
}

fn lerp_color(a: egui::Color32, b: egui::Color32, t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    egui::Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
    )
}

fn color_for(v: F, scale: F) -> egui::Color32 {
    let t = (v / scale).clamp(-1.0, 1.0) as f32;
    let white = egui::Color32::WHITE;
    let red = egui::Color32::from_rgb(220, 50, 47);
    let blue = egui::Color32::from_rgb(38, 139, 210);
    if t >= 0.0 {
        lerp_color(white, red, t)
    } else {
        lerp_color(blue, white, -t)
    }
}

fn heatmap(ui: &mut egui::Ui, label: &str, data: &[F], rows: usize, cols: usize) {
    let cell = 8.0;
    let scale = data
        .iter()
        .map(|v| v.abs())
        .fold(0.0, F::max)
        .max(1e-12);
    ui.vertical(|ui| {
        ui.label(label);
        let size = egui::vec2(cols as f32 * cell, rows as f32 * cell);
        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
        let painter = ui.painter_at(rect);
        for r in 0..rows {
            for c in 0..cols {
                let v = data[r * cols + c];
                let color = color_for(v, scale);
                let min = egui::pos2(
                    rect.min.x + c as f32 * cell,
                    rect.min.y + r as f32 * cell,
                );
                painter.rect_filled(
                    egui::Rect::from_min_size(min, egui::vec2(cell, cell)),
                    0.0,
                    color,
                );
            }
        }
    });
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.training {
            self.run_steps(self.train_steps);
            ui.ctx().request_repaint();
        }

        egui::Panel::top("controls").show(ui, |ui| {
            ui.horizontal(|ui| {
                let train_label = if self.training { "Pause" } else { "Train" };
                if ui.button(train_label).clicked() {
                    self.training = !self.training;
                    if !self.training {
                        self.recompute_boundary();
                    }
                }
                if ui.button("Step").clicked() {
                    self.run_steps(self.train_steps);
                    self.recompute_boundary();
                }
                if ui.button("Reset").clicked() {
                    self.reset();
                }
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.train_steps, 1..=5000).text("steps"),
                );
                ui.add(egui::Slider::new(&mut self.net.lr, 0.0..=1.0).text("lr"));
                ui.checkbox(&mut self.log_scale, "log loss");
                ui.separator();
                ui.label(format!(
                    "loss {:.4}   delta {:.2e}   acc {:.3}",
                    self.last_loss, self.last_delta, self.last_acc
                ));
                ui.separator();
                match self.probs {
                    Some(p) => ui.label(format!(
                        "click ({:.2},{:.2}) -> class {}  p=[{:.2},{:.2},{:.2}]",
                        self.clicked.unwrap()[0],
                        self.clicked.unwrap()[1],
                        argmax(&p),
                        p[0],
                        p[1],
                        p[2]
                    )),
                    None => ui.label("click the spiral to predict"),
                };
            });
        });

        egui::Panel::bottom("weights")
            .resizable(true)
            .default_size(230.0)
            .show(ui, |ui| {
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for i in 0..self.net.ws.len() {
                            let (in_dim, out_dim) = self.net.dimensions[i];
                            heatmap(
                                ui,
                                &format!("w{} ({}×{})", i + 1, in_dim, out_dim),
                                &self.net.ws[i],
                                in_dim,
                                out_dim,
                            );
                            heatmap(
                                ui,
                                &format!("b{} (1×{})", i + 1, out_dim),
                                &self.net.bs[i],
                                1,
                                out_dim,
                            );
                        }
                    });
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.columns(2, |cols| {
                let [xmin, xmax, ymin, ymax] = self.bounds;
                Plot::new("spiral")
                    .data_aspect(1.0)
                    .legend(Legend::default())
                    .show(&mut cols[0], |plot_ui| {
                        let hm = Heatmap::new(self.boundary.clone(), GRID)
                            .at(PlotPoint::new(xmin, ymin))
                            .size((xmax - xmin) as f32, (ymax - ymin) as f32)
                            .show_labels(false)
                            .custom_mapping(Box::new(|v| class_color(v as usize, 70)));
                        plot_ui.add(hm);

                        for c in 0..3 {
                            let pts = PlotPoints::from(self.data_by_class[c].clone());
                            plot_ui.add(
                                Points::new(format!("class {c}"), pts)
                                    .radius(3.0)
                                    .filled(true)
                                    .color(class_color(c, 255)),
                            );
                        }

                        if let Some([cx, cy]) = self.clicked {
                            let pts = PlotPoints::from(vec![[cx, cy]]);
                            plot_ui.add(
                                Points::new("clicked", pts)
                                    .radius(7.0)
                                    .shape(MarkerShape::Plus)
                                    .filled(false)
                                    .color(egui::Color32::BLACK),
                            );
                        }

                        let response = plot_ui.response();
                        if response.clicked() {
                            if let Some(pp) = plot_ui.pointer_coordinate() {
                                self.clicked = Some([pp.x, pp.y]);
                                self.probs = Some(predict(&self.net, pp.x, pp.y));
                            }
                        }
                    });

                let loss_pts: Vec<[f64; 2]> = self
                    .loss_hist
                    .iter()
                    .map(|&[s, l]| {
                        let v = if self.log_scale { l.max(1e-12).log10() } else { l };
                        [s, v]
                    })
                    .collect();
                Plot::new("loss")
                    .height(280.0)
                    .legend(Legend::default())
                    .show(&mut cols[1], |plot_ui| {
                        plot_ui.line(
                            Line::new("loss", PlotPoints::from(loss_pts))
                                .color(egui::Color32::from_rgb(0, 130, 200)),
                        );
                    });

                let acc_pts: Vec<[f64; 2]> = self.acc_hist.clone();
                Plot::new("accuracy")
                    .height(280.0)
                    .legend(Legend::default())
                    .show(&mut cols[1], |plot_ui| {
                        plot_ui.line(
                            Line::new("accuracy", PlotPoints::from(acc_pts))
                                .color(egui::Color32::from_rgb(0, 160, 90)),
                        );
                    });
            });
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "gptshin trainer",
        options,
        Box::new(|_cc| Ok(Box::new(App::new()))),
    )
}
