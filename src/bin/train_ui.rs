// src/bin/train_ui.rs
// A visualization dashboard for the Net in src/net.rs.
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use gptshin::net::Net;
use gptshin::ops::*;
use gptshin::rng::Rng;
use gptshin::toy::make_spiral;

struct App {
    net: Net,
    x: Vec<F>,
    y: Vec<usize>,
    n: usize,
    loss_hist: Vec<[f64; 2]>,
    acc_hist: Vec<[f64; 2]>,
    playing: bool,
    steps_per_frame: u32,
    log_scale: bool,
    seed: u64,
    prev_loss: Option<F>,
    last_loss: F,
    last_delta: F,
    last_acc: F,
}

impl App {
    fn new() -> App {
        let seed = 0;
        let mut rng = Rng::new(seed);
        let (x, y) = make_spiral(100, 3, &mut rng);
        let n = y.len();
        App {
            net: Net::new(64, &mut rng),
            x,
            y,
            n,
            loss_hist: Vec::new(),
            acc_hist: Vec::new(),
            playing: true,
            steps_per_frame: 1,
            log_scale: false,
            seed,
            prev_loss: None,
            last_loss: f64::NAN,
            last_delta: f64::NAN,
            last_acc: f64::NAN,
        }
    }

    fn step_once(&mut self) {
        let (cache, loss) = self.net.forward(&self.x, self.n, &self.y);
        let delta = self.prev_loss.map(|p| (p - loss).abs()).unwrap_or(loss);
        self.prev_loss = Some(loss);

        // recompute logits for accuracy without changing Net
        let mut logits = vec![0.0; self.n * 3];
        mm(&cache.a, &self.net.w2, &mut logits, self.n, self.net.h, 3);
        add_bias(&mut logits, &self.net.b2);
        let acc = accuracy(&logits, &self.y, 3);

        let (dw1, db1, dw2, db2) = self.net.backward(&cache, self.n, &self.x);
        self.net.learn(dw1, db1, dw2, db2);

        let step = self.loss_hist.len() as f64;
        self.loss_hist.push([step, loss]);
        self.acc_hist.push([step, acc]);
        self.last_loss = loss;
        self.last_delta = delta;
        self.last_acc = acc;
    }

    fn reset(&mut self) {
        self.seed += 1;
        let mut rng = Rng::new(self.seed);
        let (x, y) = make_spiral(100, 3, &mut rng);
        self.n = y.len();
        self.x = x;
        self.y = y;
        self.net = Net::new(64, &mut rng);
        self.loss_hist.clear();
        self.acc_hist.clear();
        self.prev_loss = None;
        self.last_loss = f64::NAN;
        self.last_delta = f64::NAN;
        self.last_acc = f64::NAN;
        self.playing = true;
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
        if self.playing {
            for _ in 0..self.steps_per_frame {
                self.step_once();
            }
            ui.ctx().request_repaint();
        }

        egui::Panel::top("controls").show(ui, |ui| {
            ui.horizontal(|ui| {
                let play_label = if self.playing { "Pause" } else { "Play" };
                if ui.button(play_label).clicked() {
                    self.playing = !self.playing;
                }
                if ui.button("Step").clicked() {
                    self.step_once();
                }
                if ui.button("Reset").clicked() {
                    self.reset();
                }
                ui.separator();
                ui.add(
                    egui::Slider::new(&mut self.steps_per_frame, 1..=200).text("steps/frame"),
                );
                ui.add(egui::Slider::new(&mut self.net.lr, 0.0..=1.0).text("lr"));
                ui.checkbox(&mut self.log_scale, "log loss");
                ui.separator();
                ui.label(format!(
                    "loss {:.4}   delta {:.2e}   acc {:.3}",
                    self.last_loss, self.last_delta, self.last_acc
                ));
            });
        });

        egui::Panel::bottom("weights")
            .resizable(true)
            .default_size(230.0)
            .show(ui, |ui| {
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let h = self.net.h;
                        let w1 = self.net.w1.clone();
                        heatmap(ui, "w1 (2×h)", &w1, 2, h);

                        let w2 = &self.net.w2;
                        let w2t: Vec<F> = (0..3)
                            .flat_map(|o| (0..h).map(move |hi| w2[hi * 3 + o]))
                            .collect();
                        heatmap(ui, "w2ᵀ (3×h)", &w2t, 3, h);

                        let b1 = self.net.b1.clone();
                        heatmap(ui, "b1 (1×h)", &b1, 1, h);

                        let b2 = self.net.b2.clone();
                        heatmap(ui, "b2 (1×3)", &b2, 1, 3);
                    });
                });
            });

        egui::CentralPanel::default().show(ui, |ui| {
            ui.columns(2, |cols| {
                let loss_pts: Vec<[f64; 2]> = self
                    .loss_hist
                    .iter()
                    .map(|&[s, l]| {
                        let v = if self.log_scale { l.max(1e-12).log10() } else { l };
                        [s, v]
                    })
                    .collect();
                Plot::new("loss")
                    .height(320.0)
                    .legend(Legend::default())
                    .show(&mut cols[0], |plot_ui| {
                        plot_ui.line(
                            Line::new(
                                "loss",
                                PlotPoints::from(loss_pts),
                            )
                            .color(egui::Color32::from_rgb(0, 130, 200)),
                        );
                    });

                let acc_pts: Vec<[f64; 2]> = self.acc_hist.clone();
                Plot::new("accuracy")
                    .height(320.0)
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
