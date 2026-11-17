"""Turn OpenWebText parquet shards into flat uint16 token files.

Pipeline per shard (one worker process each):
  1. read the documents' text from parquet
  2. normalize to the text8 alphabet: lowercase, digits spelled out,
     everything outside a-z becomes a space, runs of spaces collapse to one
  3. encode with the chosen tokenizer (BPE by default, or the old text8 vocab)
  4. write the token ids as raw uint16 to data/owt_<tok>/shard_XXXXX.bin

Then the shard files are concatenated into data/owt_<tok>_train.bin, and a
held-out shard becomes data/owt_<tok>_val.bin. Training reads them with
np.memmap, so the whole dataset never has to fit in RAM.

With BPE it also writes data/text8_bpe_val.bin: the exact text8 val split
(last 10% of tokens, as in tokenizer.tokenize) encoded with BPE, so a BPE model
can be compared against the old text8 runs on the same text.

uint16 works because both vocabs (~12k tokens) are below 65,536.

Usage:  uv run python -m stage5.prepare_owt [n_train_shards] [bpe|text8]
"""

import os
import re
import sys
from array import array
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

from stage5 import tokenizer as tok
from stage5.bpe import BPE

SRC_DIR = "data/openwebtext/plain_text"
VAL_SHARD = 79  # last shard is never used for training
VAL_TOKENS = 5_000_000
MAX_WORKERS = 6  # each worker holds a shard's tokens + encode cache (~1-2 GB)

DIGITS = str.maketrans(
    {
        "0": " zero ",
        "1": " one ",
        "2": " two ",
        "3": " three ",
        "4": " four ",
        "5": " five ",
        "6": " six ",
        "7": " seven ",
        "8": " eight ",
        "9": " nine ",
    }
)
NON_ALPHA = re.compile(r"[^a-z]+")


def normalize(text: str) -> str:
    """Same cleaning idea as text8's wikifil.pl."""
    text = text.lower().translate(DIGITS)
    return NON_ALPHA.sub(" ", text).strip()


def train_file(kind: str) -> str:
    return f"data/owt_{kind}_train.bin"


def val_file(kind: str) -> str:
    return f"data/owt_{kind}_val.bin"


TEXT8_BPE_VAL = "data/text8_bpe_val.bin"

# Per-worker state, set by _init_worker.
_kind = "bpe"
_bpe: BPE | None = None


def _init_worker(kind: str) -> None:
    global _kind, _bpe
    _kind = kind
    if kind == "bpe":
        _bpe = BPE.load()
    else:
        tok.build_vocab("text8", "tokens.txt")


def _encode_doc_text8(doc: str, ids: array, cache: dict[str, list[int]]) -> None:
    space = tok.stoi[tok.SPACE_TOKEN]
    for word in doc.split(" "):
        pieces = cache.get(word)
        if pieces is None:
            pieces = tok._encode_word(word)
            cache[word] = pieces
        ids.extend(pieces)
        ids.append(space)  # also separates consecutive documents


def encode_shard(shard: int) -> tuple[int, str, int]:
    src = os.path.join(SRC_DIR, f"train-{shard:05d}-of-00080.parquet")
    out_dir = f"data/owt_{_kind}"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"shard_{shard:05d}.bin")
    cache: dict[str, list[int]] = {}
    ids = array("H")  # uint16, compact - a Python list would need ~10x the RAM

    # Stream the shard in batches: loading a whole shard as Python strings costs
    # ~2 GB per worker, and 16 workers at once ran a 30 GB machine out of RAM.
    pf = pq.ParquetFile(src)
    for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
        for raw in batch.column("text").to_pylist():
            doc = normalize(raw)
            if not doc:
                continue
            if _bpe is not None:
                # Leading space: the doc's first word becomes " word", the same
                # token it would be mid-text. It also separates consecutive docs.
                ids.extend(_bpe.encode(" " + doc))
            else:
                _encode_doc_text8(doc, ids, cache)

    with open(out, "wb") as f:
        ids.tofile(f)
    return shard, out, len(ids)


def write_text8_val() -> None:
    """Encode the exact text8 val split (same boundary as tokenizer.tokenize)."""
    tok.build_vocab("text8", "tokens.txt")
    with open("text8", "r", encoding="latin-1") as f:
        text = f.read()
    old_ids = tok.encode(text)
    n = int(0.9 * len(old_ids))
    offset = len(tok.decode(old_ids[:n]))  # char where the old val split starts
    val_text = text[offset:]
    ids = np.array(BPE.load().encode(val_text), dtype=np.uint16)
    ids.tofile(TEXT8_BPE_VAL)
    print(f"text8 val: {len(val_text):,} chars -> {len(ids):,} tokens -> {TEXT8_BPE_VAL}")


def main() -> None:
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    kind = sys.argv[2] if len(sys.argv) > 2 else "bpe"
    assert kind in ("bpe", "text8"), f"unknown tokenizer {kind!r}"

    shards = list(range(n_train)) + [VAL_SHARD]
    with Pool(
        min(len(shards), os.cpu_count() or 1, MAX_WORKERS),
        initializer=_init_worker,
        initargs=(kind,),
    ) as pool:
        results = {}
        for shard, path, n in pool.imap_unordered(encode_shard, shards):
            results[shard] = (path, n)
            print(f"shard {shard:>2}: {n:>12,} tokens", flush=True)

    total = 0
    with open(train_file(kind), "wb") as f:
        for shard in range(n_train):
            path, n = results[shard]
            with open(path, "rb") as src:
                f.write(src.read())
            total += n
    print(f"train: {total:,} tokens -> {train_file(kind)}")

    val_path, _ = results[VAL_SHARD]
    val = np.fromfile(val_path, dtype=np.uint16)[:VAL_TOKENS]
    val.tofile(val_file(kind))
    print(f"val:   {len(val):,} tokens -> {val_file(kind)}")

    if kind == "bpe":
        write_text8_val()


if __name__ == "__main__":
    main()
