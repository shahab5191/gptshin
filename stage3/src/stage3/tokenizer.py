from torch import Tensor, tensor

stoi: dict[str, int] = {}
itos: dict[int, str] = {}


def _build_vocab(input: str) -> None:
    chars = sorted(set(input))
    for i, c in enumerate(chars):
        stoi[c] = i
        itos[i] = c


def _ensure_vocab(input: str) -> None:
    if not stoi or not itos:
        _build_vocab(input)


def encode(input: str) -> Tensor:
    _ensure_vocab(input)
    return tensor([stoi[c] for c in input])


def decode(ids: Tensor | list[int]) -> str:
    if isinstance(ids, Tensor):
        ids = ids.tolist()
    return "".join(itos[i] for i in ids)


def vocab_size() -> int:
    return len(stoi)


def make_pairs(data: Tensor, block_size: int) -> tuple[Tensor, Tensor]:
    x = data.unfold(0, block_size, 1)[:-1]
    y = data[block_size:]
    return x, y


def tokenize(input: str) -> tuple[Tensor, Tensor]:
    data = encode(input)
    n = int(0.9 * len(data))
    return data[:n], data[n:]
