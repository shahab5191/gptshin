import os

from torch import Tensor, tensor

SPACE_TOKEN = " "

stoi: dict[str, int] = {}
itos: dict[int, str] = {}


def build_vocab(data_file: str, token_file: str, min_count: int = 5) -> None:
    if os.path.exists(token_file):
        with open(token_file, "r", encoding="latin-1") as f:
            tokens = f.read().splitlines()
    else:
        with open(data_file, "r", encoding="latin-1") as f:
            data = f.read()
        tokens, _ = _extract_tokens(data, min_count)
        tokens = sorted(set(tokens) | {SPACE_TOKEN})
        with open(token_file, "w", encoding="latin-1") as f:
            f.writelines(f"{token}\n" for token in tokens)
    tokens = sorted(set(tokens) | {SPACE_TOKEN})
    for i, token in enumerate(tokens):
        stoi[token] = i
        itos[i] = token


def _extract_lookup_table(input: str) -> dict[int, dict[str, int]]:
    n = len(input)

    cursor = 0
    lookup: dict[int, dict[str, int]] = {}
    while cursor < n:
        while cursor < n and input[cursor].isspace():
            cursor += 1
        start = cursor
        while cursor < n and not input[cursor].isspace():
            cursor += 1
        word = input[start:cursor]
        t_len = cursor - start
        lookup.setdefault(t_len, {})
        lookup[t_len][word] = lookup[t_len].get(word, 0) + 1
    return lookup


def _extract_tokens(input: str, min_count: int = 1) -> tuple[set[str], int]:
    lookup = _extract_lookup_table(input)
    if not lookup:
        return set(), 0
    frequent = {
        word
        for length, words in lookup.items()
        if length >= 2
        for word, count in words.items()
        if count >= min_count
    }
    tokens: set[str] = set()
    memo: dict[str, bool] = {}

    def composable(word: str) -> bool:
        if word in memo:
            return memo[word]
        for k in range(1, len(word)):
            prefix, suffix = word[:k], word[k:]
            if prefix in frequent and (suffix in frequent or composable(suffix)):
                memo[word] = True
                return True
        memo[word] = False
        return False

    for length in sorted(lookup):
        if length < 2:
            continue
        for word in lookup[length]:
            if not composable(word):
                tokens.add(word)
    tokens.update(lookup.get(1, set()))
    return tokens, max(lookup)


def _encode_word(word: str) -> list[int]:
    ids: list[int] = []
    i = 0
    n = len(word)
    while i < n:
        for j in range(n, i, -1):
            sub = word[i:j]
            if sub in stoi:
                ids.append(stoi[sub])
                i = j
                break
        else:
            raise ValueError(f"cannot encode {word!r} at index {i}")
    return ids


def encode(text: str) -> list[int]:
    ids: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] == " ":
            ids.append(stoi[SPACE_TOKEN])
            i += 1
            continue
        j = i
        while j < n and text[j] != " ":
            j += 1
        ids.extend(_encode_word(text[i:j]))
        i = j
    return ids


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


def vocab_size() -> int:
    return len(stoi)


def make_pairs(data: Tensor, block_size: int) -> tuple[Tensor, Tensor]:
    x = data.unfold(0, block_size, 1)[:-1]
    y = data[block_size:]
    return x, y


def tokenize(text: str) -> tuple[Tensor, Tensor]:
    data = tensor(encode(text))
    n = int(0.9 * len(data))
    return data[:n], data[n:]


if __name__ == "__main__":
    build_vocab("text8", "tokens.txt")
