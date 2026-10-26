"""Web UI (Gradio) for the stage2 spiral trainer.

Mirrors the Rust ``train_ui.rs`` dashboard: live loss/accuracy, a clickable
decision boundary, and per-layer weight/bias heatmaps.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; Gradio serves the figures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import gradio as gr

from stage2.model import Mlp
from stage2.spiral import make_spiral

GRID = 60
DIMS = [2, 64, 64, 32, 16, 8, 3]
CLASS_COLORS = {"0": "#dc322f", "1": "#00a05a", "2": "#268bd2"}


class Trainer:
    """Stateful trainer: holds model + data + optimizer across UI interactions."""

    def __init__(self, dims: list[int] = DIMS, seed: int = 0, lr: float = 0.1) -> None:
        self.dims = list(dims)
        self.seed = seed
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.running = False
        self._init()

    def _init(self) -> None:
        torch.manual_seed(self.seed)
        self.model = Mlp(self.dims).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        x, y = make_spiral(100, 3, seed=self.seed)
        self.X = torch.from_numpy(x).float().to(self.device)
        self.y = torch.from_numpy(y).long().to(self.device)
        self.losses: list[float] = []
        self.accs: list[float] = []
        self.step_count = 0

    def train_steps(self, n: int) -> None:
        self.model.train()
        for _ in range(int(n)):
            self.optimizer.zero_grad()
            logits = self.model(self.X)
            loss = torch.nn.functional.cross_entropy(logits, self.y)
            loss.backward()
            self.optimizer.step()
            self.losses.append(float(loss.item()))
            self.accs.append(float((logits.argmax(1) == self.y).float().mean().item()))
            self.step_count += 1

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr

    def reset(self) -> None:
        self.seed += 1
        self._init()

    def predict(self, x: float, y: float) -> tuple[int, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            inp = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)
            probs = torch.softmax(self.model(inp), dim=1)[0].cpu().numpy()
        return int(probs.argmax()), probs

    def decision_boundary(self, grid: int = GRID) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        xn = self.X.cpu().numpy()
        xmin, xmax = xn[:, 0].min(), xn[:, 0].max()
        ymin, ymax = xn[:, 1].min(), xn[:, 1].max()
        px, py = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
        xs = np.linspace(xmin - px, xmax + px, grid)
        ys = np.linspace(ymin - py, ymax + py, grid)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        with torch.no_grad():
            pred = self.model(torch.from_numpy(pts).float().to(self.device)).argmax(1)
        return gx, gy, pred.cpu().numpy().reshape(grid, grid)


trainer = Trainer()


# --- data helpers (return values for the plot components) ---

def status_md() -> str:
    if not trainer.losses:
        return "No training steps yet. Click **Train** or **Step**."
    return (
        f"**step {trainer.step_count}** — loss {trainer.losses[-1]:.4f} — "
        f"acc {trainer.accs[-1]:.3f}"
    )


def loss_df() -> pd.DataFrame:
    return pd.DataFrame({"step": range(len(trainer.losses)), "loss": trainer.losses})


def acc_df() -> pd.DataFrame:
    return pd.DataFrame({"step": range(len(trainer.accs)), "acc": trainer.accs})


def boundary_df() -> pd.DataFrame:
    gx, gy, pred = trainer.decision_boundary()
    return pd.DataFrame(
        {"x": gx.ravel(), "y": gy.ravel(), "class": pred.ravel().astype(int).astype(str)}
    )


def data_df() -> pd.DataFrame:
    xn = trainer.X.cpu().numpy()
    yn = trainer.y.cpu().numpy()
    return pd.DataFrame(
        {"x": xn[:, 0], "y": xn[:, 1], "class": yn.astype(int).astype(str)}
    )


def weights_fig() -> plt.Figure:
    layers = list(trainer.model.layers)
    n = len(layers)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3 * n))
    for i, layer in enumerate(layers):
        w = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()
        im = axes[i, 0].imshow(w, aspect="auto", cmap="RdBu_r")
        axes[i, 0].set_title(f"W{i + 1}  {w.shape}")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046)
        axes[i, 1].bar(range(len(b)), b)
        axes[i, 1].set_title(f"b{i + 1}  ({len(b)})")
    fig.tight_layout()
    return fig


# --- callbacks ---

def tick(steps: float, lr: float) -> tuple:
    if trainer.running:
        trainer.set_lr(lr)
        trainer.train_steps(steps)
    return (status_md(), loss_df(), acc_df(), boundary_df(), weights_fig())


def do_step(steps: float, lr: float) -> tuple:
    trainer.set_lr(lr)
    trainer.train_steps(steps)
    return (status_md(), loss_df(), acc_df(), boundary_df(), weights_fig())


def do_reset() -> tuple:
    trainer.reset()
    return (status_md(), loss_df(), acc_df(), boundary_df(), weights_fig(), "")


def toggle() -> tuple:
    trainer.running = not trainer.running
    return gr.update(active=trainer.running), ("Pause" if trainer.running else "Train")


def on_select(evt: gr.SelectData) -> str:
    if evt.index is None:
        return ""
    x, y = evt.index
    cls, probs = trainer.predict(float(x), float(y))
    p = ", ".join(f"{v:.3f}" for v in probs)
    return f"Clicked **({x:.3f}, {y:.3f})** → class **{cls}** — probs [{p}]"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="gptshin stage2") as demo:
        gr.Markdown("# gptshin — stage2 (PyTorch)")
        status = gr.Markdown(status_md())

        with gr.Row():
            btn_train = gr.Button("Train", variant="primary")
            btn_step = gr.Button("Step")
            btn_reset = gr.Button("Reset")
            steps_slider = gr.Slider(1, 5000, value=100, step=1, label="steps")
            lr_slider = gr.Slider(0.0001, 1.0, value=0.1, label="lr")

        with gr.Row():
            loss_plot = gr.LinePlot(loss_df(), x="step", y="loss", title="Loss")
            acc_plot = gr.LinePlot(acc_df(), x="step", y="acc", title="Accuracy")

        with gr.Row():
            boundary_plot = gr.ScatterPlot(
                boundary_df(),
                x="x",
                y="y",
                color="class",
                color_map=CLASS_COLORS,
                title="Decision boundary (click to predict)",
                height=400,
            )
            data_plot = gr.ScatterPlot(
                data_df(),
                x="x",
                y="y",
                color="class",
                color_map=CLASS_COLORS,
                title="Spiral data (true labels)",
                height=400,
            )

        click_result = gr.Markdown("Click the decision boundary to see the model's prediction.")

        with gr.Row():
            weights_plot = gr.Plot(weights_fig(), label="Weights & biases")

        timer = gr.Timer(0.5, active=False)

        outputs = [status, loss_plot, acc_plot, boundary_plot, weights_plot]

        timer.tick(tick, inputs=[steps_slider, lr_slider], outputs=outputs)
        btn_step.click(do_step, inputs=[steps_slider, lr_slider], outputs=outputs)
        btn_reset.click(do_reset, outputs=outputs + [click_result])
        btn_train.click(toggle, outputs=[timer, btn_train])
        boundary_plot.select(on_select, outputs=[click_result])

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860)
