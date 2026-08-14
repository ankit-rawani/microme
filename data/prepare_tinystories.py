"""Tokenize TinyStories -> uint16 .bin shards (plan §0, nanoGPT style).

Each story prefixed with <|bos|> as document delimiter, all concatenated into
one flat token stream per split, written incrementally so RAM stays flat.
"""
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "tinystories-bpe.json"))
BOS = TOK.token_to_id("<|bos|>")
OUTDIR = ROOT / "data" / "tinystories"
BATCH = 20_000


def write_split(split, out):
    n = 0
    with open(out, "wb") as f:
        buf, texts = [], []
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        def flush():
            nonlocal n
            for enc in TOK.encode_batch(texts):
                buf.extend((BOS, *enc.ids))
            arr = np.array(buf, dtype=np.uint16)
            arr.tofile(f)
            n += arr.size
            buf.clear(); texts.clear()
        for ex in ds:
            texts.append(ex["text"])
            if len(texts) >= BATCH:
                flush()
        if texts:
            flush()
    print(f"{split}: {n:,} tokens -> {out}")
    return n


if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    write_split("validation", OUTDIR / "val.bin")
    write_split("train", OUTDIR / "train.bin")
