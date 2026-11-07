stoi: dict[str, int] = {}
itos: dict[int, str] = {}


def _extract_lookup_table(input: str) -> dict[int, set[str]]:
    n = len(input)

    cursor = 0
    lookup: dict[int, set[str]] = {}
    while cursor < n:
        while cursor < n and input[cursor].isspace():
            cursor += 1
        start = cursor
        while cursor < n and not input[cursor].isspace():
            cursor += 1
        t_len = cursor - start
        lookup.setdefault(t_len, set()).add(input[start:cursor])
    return lookup


def _extract_tokens(input: str) -> tuple[set[str], int]:
    lookup = _extract_lookup_table(input)
    if not lookup:
        return set(), 0
    tokens: set[str] = set()
    memo: dict[str, bool] = {}

    def composable(word: str) -> bool:
        if word in memo:
            return memo[word]
        for k in range(1, len(word)):
            prefix, suffix = word[:k], word[k:]
            if prefix in tokens and (suffix in tokens or composable(suffix)):
                memo[word] = True
                return True
        memo[word] = False
        return False

    for length in sorted(lookup):
        for word in lookup[length]:
            if length == 1 or not composable(word):
                tokens.add(word)
    return tokens, max(lookup)
