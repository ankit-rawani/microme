"""Render databricks-dolly-15k -> packed chat tokens + assistant-only mask.

Short, human-written, factual instruction/response pairs (median ~50 resp
tokens) — the anti-rambling polish set. Single-turn; when a 'context' is given
it's prepended to the user turn (grounded/closed-book QA).
Same bin format as prepare_smoltalk.py.
"""
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "fineweb-bpe.json"))
S = {k: TOK.token_to_id(f"<|{k}|>") for k in ("bos", "user", "assistant", "end")}
OUTDIR = ROOT / "data" / "dolly"
VAL = 500


def render(user, asst):
    ids, mask = [S["bos"], S["user"]], [0, 0]
    uc = TOK.encode(user).ids
    ids += uc + [S["end"]]; mask += [0] * (len(uc) + 1)
    ids.append(S["assistant"]); mask.append(0)
    ac = TOK.encode(asst).ids
    ids += ac + [S["end"]]; mask += [1] * (len(ac) + 1)   # assistant content + stop
    return ids, mask


def write(rows, stem):
    tf = open(OUTDIR / f"{stem}.bin", "wb"); mf = open(OUTDIR / f"{stem}.mask", "wb")
    n = 0
    for r in rows:
        user = (r["context"].strip() + "\n\n" + r["instruction"]) if r.get("context") else r["instruction"]
        ids, mask = render(user, r["response"])
        np.array(ids, dtype=np.uint16).tofile(tf)
        np.array(mask, dtype=np.uint8).tofile(mf)
        n += len(ids)
    tf.close(); mf.close(); print(f"{stem}: {n:,} tokens")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=0)
    write(ds.select(range(VAL)), "val")
    write(ds.select(range(VAL, len(ds))), "train")


if __name__ == "__main__":
    main()
