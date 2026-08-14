"""M5 — chat SFT on SmolTalk (plan §7.1, §11 gate: follows multi-turn chat).

Loads the micro_125m base checkpoint, fine-tunes on packed chat data with
assistant-only masked loss. AdamW at a low lr (SFT is short); bf16 + compile;
resumable. Eval = actually generate a reply and check it stops at <|end|>.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, PRESETS
from train import log_report

ROOT = Path(__file__).parent
BASE = ROOT / "runs" / "micro_125m_muon" / "ckpt.pt"
DATA = ROOT / "data" / "smoltalk"
OUT = ROOT / "runs" / "micro_125m_sft"
TOKF = ROOT / "tokenizer" / "fineweb-bpe.json"

PRESET = "micro_125m"
BATCH, ACCUM = 4, 16          # eff 65,536 tok/step
EPOCHS = 1                    # 391M diverse tokens is ample for 125M; avoids overfit. Resume for more.
LR, MIN_FRAC, WARMUP = 2e-4, 0.1, 100
EVAL_EVERY, CKPT_SECS = 200, 1800
PROMPTS = ["What is the capital of France?",
           "Explain photosynthesis in one sentence.",
           "Write a haiku about the ocean."]


def get_batch(tok, msk, B, ctx, device):
    ix = torch.randint(len(tok) - ctx - 1, (B,))
    x = torch.stack([torch.from_numpy(tok[i:i + ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tok[i + 1:i + 1 + ctx].astype(np.int64)) for i in ix])
    m = torch.stack([torch.from_numpy(msk[i + 1:i + 1 + ctx].astype(np.float32)) for i in ix])
    return (x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
            m.pin_memory().to(device, non_blocking=True))


def lr_mult(it, total):
    w = max(10, min(WARMUP, total // 10))            # auto-scale warmup for short runs
    if it < w:
        return (it + 1) / w
    frac = (it - w) / max(1, total - w)
    return MIN_FRAC + (1 - MIN_FRAC) * (1 - frac)   # linear decay to floor


@torch.no_grad()
def chat(raw, tok, S, user, device, max_new=120):
    ids = [S["bos"], S["user"]] + tok.encode(user).ids + [S["end"], S["assistant"]]
    raw.eval()
    out = raw.generate(torch.tensor([ids], device=device), max_new, temperature=0.7, top_k=40)[0].tolist()
    raw.train()
    gen = out[len(ids):]
    stopped = S["end"] in gen
    if stopped:
        gen = gen[:gen.index(S["end"])]
    return tok.decode(gen), stopped


@torch.no_grad()
def val_loss(model, tok, msk, ctx, device, iters=40):
    model.eval()
    tot = 0.0
    for _ in range(iters):
        x, y, m = get_batch(tok, msk, BATCH, ctx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, m)
        tot += loss.item()
    model.train()
    return tot / iters


def main():
    device = "cuda"
    torch.manual_seed(1337); torch.set_float32_matmul_precision("high")
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(TOKF))
    S = {k: tok.token_to_id(f"<|{k}|>") for k in ("bos", "user", "assistant", "end", "system")}
    cfg = PRESETS[PRESET]

    tr = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    trm = np.memmap(DATA / "train.mask", dtype=np.uint8, mode="r")
    va = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    vam = np.memmap(DATA / "val.mask", dtype=np.uint8, mode="r")
    eff = BATCH * ACCUM * cfg.ctx
    total = EPOCHS * len(tr) // eff
    ckpt = OUT / "ckpt.pt"

    raw = GPT(cfg).to(device)
    start = 0
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device)
        raw.load_state_dict(ck["model"]); start = ck["iter"] + 1
        opt = torch.optim.AdamW(raw.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
        opt.load_state_dict(ck["opt"]); print(f"resumed SFT at {start}")
    else:
        base = torch.load(BASE, map_location=device)
        raw.load_state_dict(base["model"]); print(f"loaded base (val {base.get('best_val')}) from {BASE}")
        opt = torch.optim.AdamW(raw.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    model = torch.compile(raw)
    print(f"SFT {PRESET}: {len(tr):,} chat tok | eff {eff:,}/step | {total} iters (~{EPOCHS} epochs)")

    t0 = last = time.time()
    for it in range(start, total):
        lr = LR * lr_mult(it, total)
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tl = 0.0
        for _ in range(ACCUM):
            x, y, m = get_batch(tr, trm, BATCH, cfg.ctx, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, m)
            (loss / ACCUM).backward(); tl += loss.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        opt.step()
        if it % 50 == 0:
            print(f"iter {it}/{total} | loss {tl:.3f} | lr {lr:.2e} | {time.time()-t0:.0f}s", flush=True)
        if it % EVAL_EVERY == 0 or it == total - 1:
            vl = val_loss(model, va, vam, cfg.ctx, device)
            reply, stopped = chat(raw, tok, S, PROMPTS[0], device)
            print(f"  eval {it}: val {vl:.3f} | stop:{stopped} | reply[{PROMPTS[0]}]: {reply[:160]!r}", flush=True)
        if time.time() - last > CKPT_SECS or it == total - 1:
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "iter": it}, ckpt)
            last = time.time(); print(f"  ckpt @ {it}", flush=True)

    samples = [(p,) + chat(raw, tok, S, p, device) for p in PROMPTS]
    log_report(f"- **M5 SFT {PRESET}** ({datetime.now():%Y-%m-%d %H:%M}) | {total} iters | val {vl:.3f}\n"
               + "\n".join(f"  - Q: {p}\n    A[stop:{s}]: {r[:200]!r}" for p, r, s in samples))
    print("done:", f"{(time.time()-t0)/60:.1f} min | val {vl:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))    # init checkpoint (base model, or a prior SFT to continue)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--accum", type=int, default=ACCUM)
    a = ap.parse_args()
    BASE, DATA, OUT = Path(a.base), Path(a.data), Path(a.out)
    EPOCHS, LR, ACCUM = a.epochs, a.lr, a.accum
    main()
