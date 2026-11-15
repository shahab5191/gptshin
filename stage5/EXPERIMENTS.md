# stage5 — Experiments log

## Architecture (current, both runs)

Self-attention MLP language model:
- token embedding + positional embedding → `[B, T, D]`
- per-position query attention (`q = W_q(x)`, `w = q·xᵀ`, softmax, weighted sum)
- causal mask (each position attends only to past)
- residual MLP: `in_proj → [blocks] → out_proj`

## Run 1 — BASELINE (training now)

Per-position prediction (predicts the next token at *every* position).

Config (`src/stage5/__init__.py`):
```
BLOCK_SIZE = 512       (context)
EMBED_DIM = 512
HIDDEN_W   = 400
HIDDEN_C   = 20        (residual blocks)
RESIDUAL   = True
LEARNING_RATE = 1e-4
STEPS      = 100_000
batch (train & eval) = 64
```

Result (fill in when done):
```
final val loss = ____  (ppl ____)
```

## Run 2 — TOMORROW: collapsed (1-token prediction) + bigger context

Goal: compare "bigger context" vs "per-position signal". Memory-limited GPU (15.5 GB),
time is not the constraint → collapse to last-token prediction, keep healthy batch,
spend freed memory on context.

Config:
```
prediction = last token only   (collapsed)
batch = 128
BLOCK_SIZE = 4096   (try 8192 if memory allows)
HIDDEN_W / HIDDEN_C / RESIDUAL = same as baseline (to isolate the variable)
```

### Code changes needed for collapse

1. `model.py` forward — compute ONLY the last position's attention (O(T), not O(T²)):
   ```python
   x = self.embed(x) + self.pos_embed.weight        # [B, T, D]
   q = self.W_q(x[:, -1])                            # [B, D]  last position only
   w = torch.einsum("bd,bsd->bs", q, x)              # [B, T]  one attention row
   w = w / (self.embed_dim ** 0.5)
   w = w.softmax(dim=-1)
   x = torch.einsum("bs,bsd->bd", w, x)              # [B, D]  weighted sum
   x = F.relu(self.in_proj(x))                       # [B, H]
   ... blocks ...
   x = self.out_proj(x)                              # [B, D]
   logits = F.linear(x, self.embed.weight)           # [B, V]
   ```
   (causal mask no longer needed — the last position attends to everything)

2. `tokenizer.py` make_pairs — revert to single target per window:
   ```python
   n_windows = (len(data) - block_size) // stride
   x = data.unfold(0, block_size, stride)[:n_windows]
   y = data[block_size : block_size + n_windows * stride : stride]
   ```

3. `train.py` loss — logits are `[B, V]`, targets `[B]` (remove the `.reshape(-1, ...)`):
   ```python
   loss = nn.functional.cross_entropy(yhat, yb)          # train
   loss = nn.functional.cross_entropy(logits, yb, reduction="sum")  # eval, total += yb.shape[0]
   ```

4. `predict.py` — `model(x).argmax(dim=1)`

5. `__init__.py` generation — `probs = logits[0].softmax(dim=-1)`

### Comparison

```
Run 1 (per-position, T=512,   batch 64):  val ppl ____
Run 2 (collapsed,    T=4096,  batch 128): val ppl ____
```

Watch: train & val loss. Also note total wall-clock steps/speed for each.
If Run 2 underperforms → context is NOT the bottleneck; try wider HIDDEN_W instead
(keep collapsed + batch 128, set context 512, HIDDEN_W 1024).
