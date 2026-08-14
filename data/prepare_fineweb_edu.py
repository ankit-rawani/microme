"""Stream FineWeb-Edu -> uint16 .bin shards (plan §6, M4 main run).

Streams HuggingFaceFW/fineweb-edu (sample-100BT), tokenizes on the fly with
the FineWeb BPE, writes a flat token stream. First VAL_DOCS documents become a
disjoint held-out val set (for the val<=3.4 gate); the rest fill train.bin up
to TRAIN_TOKENS (~3B, ~24x Chinchilla for 125M — the SmolLM overtraining recipe).

RAM stays flat (incremental writes). Disk ~6 GB.
ponytail: pure FineWeb-Edu. The §6 5% code + 5% QA mix helps *chat* behavior,
which lands at midtraining (M5) — add it there, not in the pretrain gate.
"""
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "fineweb-bpe.json"))
BOS = TOK.token_to_id("<|bos|>")
OUTDIR = ROOT / "data" / "fineweb_edu"
BATCH = 8_000
VAL_DOCS = 5_000
TRAIN_TOKENS = 3_000_000_000


def encode_batch(texts):
    out = []
    for enc in TOK.encode_batch(texts):
        out.append(BOS)
        out.extend(enc.ids)
    return np.array(out, dtype=np.uint16)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
    it = iter(ds)

    # val: first VAL_DOCS docs (disjoint from train)
    with open(OUTDIR / "val.bin", "wb") as f:
        buf, n = [], 0
        for _ in range(VAL_DOCS):
            buf.append(next(it)["text"])
            if len(buf) >= BATCH:
                a = encode_batch(buf); a.tofile(f); n += a.size; buf = []
        if buf:
            a = encode_batch(buf); a.tofile(f); n += a.size
    print(f"val: {n:,} tokens")

    # train: stream until TRAIN_TOKENS
    written = 0
    with open(OUTDIR / "train.bin", "wb") as f:
        buf = []
        for ex in it:
            buf.append(ex["text"])
            if len(buf) >= BATCH:
                a = encode_batch(buf); a.tofile(f); written += a.size; buf = []
                if written >= TRAIN_TOKENS:
                    break
                if written % (100_000_000) < BATCH * 300:  # ~ every 100M tok
                    print(f"  train: {written/1e9:.2f}B tokens")
        if buf and written < TRAIN_TOKENS:
            a = encode_batch(buf); a.tofile(f); written += a.size
    print(f"train: {written:,} tokens -> {OUTDIR/'train.bin'}")


if __name__ == "__main__":
    main()
