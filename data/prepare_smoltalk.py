"""Render SmolTalk conversations -> packed uint16 tokens + uint8 loss mask (plan §7.1).

Chat format (our special tokens):
  <|bos|> <|system|>sys<|end|> <|user|>q<|end|> <|assistant|>a<|end|> <|user|>...<|assistant|>...<|end|>

loss mask = 1 ONLY on assistant content + its closing <|end|> (SFT: learn to
respond and to stop, not to model the user's turns). Everything packed into one
flat stream; the SFT loader samples ctx-length windows and masks the loss.
"""
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent
TOK = Tokenizer.from_file(str(ROOT / "tokenizer" / "fineweb-bpe.json"))
S = {k: TOK.token_to_id(f"<|{k}|>") for k in ("bos", "user", "assistant", "end", "system")}
ROLE = {"system": S["system"], "user": S["user"], "assistant": S["assistant"]}
OUTDIR = ROOT / "data" / "smoltalk"
MAX_CONVOS = 400_000   # ~§7 "300-500k"; cap to bound time
VAL_CONVOS = 2_000


def render(messages):
    ids, mask = [S["bos"]], [0]
    for m in messages:
        asst = m["role"] == "assistant"
        content = TOK.encode(m["content"]).ids
        ids.append(ROLE.get(m["role"], S["user"])); mask.append(0)   # role marker: never loss
        ids.extend(content);                        mask.extend([1 if asst else 0] * len(content))
        ids.append(S["end"]);                       mask.append(1 if asst else 0)  # teach stop after asst
    return ids, mask


def write(convos, stem):
    tf = open(OUTDIR / f"{stem}.bin", "wb")
    mf = open(OUTDIR / f"{stem}.mask", "wb")
    ntok = 0
    for msgs in convos:
        ids, mask = render(msgs)
        np.array(ids, dtype=np.uint16).tofile(tf)
        np.array(mask, dtype=np.uint8).tofile(mf)
        ntok += len(ids)
    tf.close(); mf.close()
    print(f"{stem}: {ntok:,} tokens")
    return ntok


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
    it = iter(ds)
    val = [next(it)["messages"] for _ in range(VAL_CONVOS)]
    write(val, "val")

    def train_iter():
        for i, ex in enumerate(it):
            if i >= MAX_CONVOS:
                break
            if i % 50_000 == 0 and i:
                print(f"  train convo {i:,}")
            yield ex["messages"]
    write(train_iter(), "train")


if __name__ == "__main__":
    main()
