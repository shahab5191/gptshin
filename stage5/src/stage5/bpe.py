"""Byte-level BPE tokenizer (GPT-2 style).

Vocab layout:
    ids 0..255   the 256 raw byte values (so ANY text can be encoded)
    ids 256..    one token per learned merge, in the order it was learned

Training:  count adjacent pairs, merge the most frequent, repeat.
Encoding:  replay the merges on each chunk, earliest-learned (lowest rank) first.
Decoding:  concatenate each id's bytes, then decode UTF-8.

Before counting or encoding, text is pre-split into chunks so merges never cross
from a word into punctuation or into the next word. A space is kept at the
START of the following word (" cat"), so spaces stop being separate tokens.

Usage (train and save merges):  uv run python -m stage5.bpe [vocab_size]
"""

import heapq
import json
import re
import sys
from collections import Counter, defaultdict

# Every character falls into exactly one class, so findall loses nothing:
#   letters        = \w minus digits minus "_"
#   digits         = \d
#   everything else that is not whitespace (punctuation, "_", symbols)
#   whitespace     - "\s+(?!\S)" leaves the last space to attach to the next word
PATTERN = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+")

MERGES_FILE = "bpe_merges.json"

Pair = tuple[int, int]


def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPE:
    def __init__(self, merges: list[Pair] | None = None):
        self.merges: list[Pair] = list(merges or [])
        self.ranks: dict[Pair, int] = {p: i for i, p in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---------------------------------------------------------------- training

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = True) -> "BPE":
        # 1. Pairs never cross chunk boundaries, so we only need each distinct
        #    chunk and how often it occurs - not the whole text.
        chunk_counts = Counter(PATTERN.findall(text))
        words = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs = list(chunk_counts.values())
        if verbose:
            print(f"{len(words):,} distinct chunks")

        # 2. Count every adjacent pair, weighted by chunk frequency, and remember
        #    which chunks contain each pair (so a merge only touches those).
        pair_counts: Counter[Pair] = Counter()
        where: defaultdict[Pair, set[int]] = defaultdict(set)
        for i, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(i)

        # 3. Max-heap of (count, pair). Counts change after each merge; instead of
        #    updating entries in place we push fresh ones and skip stale ones.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[Pair] = []
        n_merges = vocab_size - 256
        while len(merges) < n_merges and heap:
            neg_c, pair = heapq.heappop(heap)
            count = pair_counts.get(pair, 0)
            if count <= 0 or count != -neg_c:
                continue  # stale heap entry
            new_id = 256 + len(merges)
            merges.append(pair)

            # 4. Apply the merge only to chunks that contain the pair, and update
            #    pair counts for just those chunks.
            changed: set[Pair] = set()
            for i in where.pop(pair):
                w, f = words[i], freqs[i]
                for q in zip(w, w[1:]):
                    pair_counts[q] -= f
                    changed.add(q)
                w = _merge(w, pair, new_id)
                words[i] = w
                for q in zip(w, w[1:]):
                    pair_counts[q] += f
                    where[q].add(i)
                    changed.add(q)
            for q in changed:
                c = pair_counts[q]
                if c > 0:
                    heapq.heappush(heap, (-c, q))
                else:
                    del pair_counts[q]

            if verbose and len(merges) % 1000 == 0:
                tok = (b"".join(cls._bytes_of(merges, t) for t in pair)).decode(
                    "utf-8", errors="replace"
                )
                print(f"merge {len(merges):>6}/{n_merges}: {tok!r:<16} count {count:,}")

        return cls(merges)

    @staticmethod
    def _bytes_of(merges: list[Pair], tid: int) -> bytes:
        if tid < 256:
            return bytes([tid])
        a, b = merges[tid - 256]
        return BPE._bytes_of(merges, a) + BPE._bytes_of(merges, b)

    # ------------------------------------------------------- encode / decode

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # the earliest-learned merge available in this chunk
            best = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1 << 62))
            rank = self.ranks.get(best)
            if rank is None:
                break  # no adjacent pair is a known merge - done
            ids = _merge(ids, best, 256 + rank)
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------- save/load

    def save(self, path: str = MERGES_FILE) -> None:
        with open(path, "w") as f:
            json.dump(self.merges, f)

    @classmethod
    def load(cls, path: str = MERGES_FILE) -> "BPE":
        with open(path) as f:
            return cls([tuple(p) for p in json.load(f)])


def main() -> None:
    import pyarrow.parquet as pq

    from stage5.prepare_owt import SRC_DIR, normalize

    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 12_288
    train_chars = 100_000_000  # same amount of text the old vocab was built from

    # Train on normalized OWT from shard 0 (a training shard - never shard 79/val).
    pf = pq.ParquetFile(f"{SRC_DIR}/train-00000-of-00080.parquet")
    parts: list[str] = []
    n = 0
    for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
        for doc in batch.column("text").to_pylist():
            s = normalize(doc)
            parts.append(s)
            n += len(s) + 1
        if n >= train_chars:
            break
    text = " ".join(parts)[:train_chars]
    print(f"training BPE on {len(text):,} chars, vocab_size={vocab_size}")

    bpe = BPE.train(text, vocab_size)
    bpe.save()
    print(f"saved {len(bpe.merges):,} merges to {MERGES_FILE}")


if __name__ == "__main__":
    main()
