"""Load the trained model and chat with it (completion-style).

The model is a base (completion-only) transformer - NOT chat-tuned - so it will
complete the prompt like web text rather than answer conversationally. That's
expected. This module is the diagnostic: observe the failure modes first, then
do supervised fine-tuning (SFT) to fix them.

Run:  uv run python -m stage5.chat
"""

import torch

from stage5.bpe import BPE
from stage5.model import Model

# These must match the checkpoint (model_bpe.pt). Best practice is to save them
# alongside the weights, but for now they mirror __init__.py's constants.
CONFIG = {
    "block_size": 8192,
    "embed_dim": 512,
    "hidden_w": 2048,
    "blocks_per_layer": 1,
    "n_layers": 8,
    "num_heads": 8,
}
MODEL_FILE = "model_bpe.pt"


def load_model(path: str = MODEL_FILE, device: str | None = None) -> tuple[Model, BPE]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bpe = BPE.load()
    model = Model(
        CONFIG["block_size"],
        bpe.vocab_size,
        CONFIG["embed_dim"],
        CONFIG["hidden_w"],
        CONFIG["blocks_per_layer"],
        CONFIG["n_layers"],
        CONFIG["num_heads"],
    )
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, bpe


@torch.no_grad()
def generate(
    model: Model,
    bpe: BPE,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
    """Autoregressively continue `prompt`, returning only the new text."""
    ids = bpe.encode(prompt)
    n_prompt = len(ids)
    context = torch.tensor([ids], dtype=torch.long, device=model.device)

    for _ in range(max_new_tokens):
        logits = model(context)[0, -1] / temperature
        if top_k is not None and top_k < logits.numel():
            top_values, top_indices = torch.topk(logits, top_k)
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(0, top_indices, top_values)
        probs = logits.softmax(dim=-1)
        nxt = torch.multinomial(probs, 1)
        context = torch.cat([context, nxt.unsqueeze(0)], dim=1)[:, -model.block_size :]

    return bpe.decode(context[0, n_prompt:].tolist())


def chat() -> None:
    model, bpe = load_model()
    print("Loaded model. Empty line quits.")
    template = (
        "The following is a conversation between a User and a helpful Assistant.\n\n"
        "User: {user}\nAssistant:"
    )
    while True:
        user = input("\nYou: ").strip()
        if not user:
            break
        prompt = template.format(user=user)
        reply = generate(model, bpe, prompt)
        print(f"Assistant: {reply}")


if __name__ == "__main__":
    chat()
