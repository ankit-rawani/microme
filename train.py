"""Training entry points.

- smoke()  : M0 gate (overfit one batch < 0.1 loss).
- run()    : real pretraining loop for micro_30m / micro_125m.
             bf16 autocast, memmap data, grad-accum, warmup+cooldown LR,
             grad clip, periodic val + story samples, resumable checkpoints.

M2 target (§6/§11): coherent 100-token TinyStories samples.
Optimizer here is AdamW (Muon is M3). Muon becomes ablation rung 1.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch init

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, PRESETS

ROOT = Path(__file__).parent
REPORT = ROOT / "REPORT.md"


def log_report(line: str):
    header = not REPORT.exists()
    with REPORT.open("a") as f:
        if header:
            f.write("# MicroMe — Run Report Card\n\n")
        f.write(line.rstrip() + "\n")


# ---------------------------------------------------------------- M0 smoke
def smoke(steps=400, lr=3e-3, target=0.1):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = GPT(PRESETS["micro_1m"]).to(dev)
    cfg = model.cfg
    B, T = 8, cfg.ctx
    x = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    t0 = time.time()
    loss = None
    for step in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")
    final, dt, ok = loss.item(), time.time() - t0, loss.item() < target
    log_report(f"- **M0 smoke** ({datetime.now():%Y-%m-%d %H:%M}) | micro_1m "
               f"{model.num_params()/1e6:.2f}M | {dev} | {steps} steps {dt:.1f}s | "
               f"loss {final:.4f} | gate<{target}: {'PASS' if ok else 'FAIL'}")
    print(f"\n{'PASS' if ok else 'FAIL'}: loss {final:.4f}")
    assert ok
    return final


# ------------------------------------------------------------ real training
@dataclass
class TrainConfig:
    preset: str = "micro_30m"
    data: str = "data/tinystories"
    out: str = "runs/micro_30m"
    tokenizer: str = "tokenizer/tinystories-bpe.json"
    optim: str = "adamw"     # "adamw" | "muon" (Muon on 2-D hidden matrices + AdamW on the rest)
    batch: int = 16          # micro-batch (sequences); 8GB-safe with 24k-vocab logits
    grad_accum: int = 8      # -> effective batch = batch*grad_accum*ctx tokens (65,536)
    max_iters: int = 8000
    warmup: int = 300
    cooldown_frac: float = 0.4   # linear decay over final 40% (§5)
    lr: float = 6e-4         # AdamW lr (embeddings/head/norms)
    muon_lr: float = 0.02    # Muon lr (shape-scaled internally, §5)
    min_frac: float = 0.1    # cooldown floor as fraction of peak lr
    wd: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_iters: int = 100
    ckpt_secs: int = 1800    # checkpoint every 30 min wall-clock (§2)
    seed: int = 1337


# Per-preset run defaults (data/tokenizer + batch/schedule sized to the model).
# micro_125m: eff batch 8*64*1024 = 524,288 tok (~§4 target); ~5700 iters ≈ 3B tok.
RUN_DEFAULTS = {
    "micro_30m":  dict(data="data/tinystories", tokenizer="tokenizer/tinystories-bpe.json",
                       batch=16, grad_accum=8, max_iters=8000, warmup=300, eval_every=250, eval_iters=100),
    "micro_125m": dict(data="data/fineweb_edu", tokenizer="tokenizer/fineweb-bpe.json",
                       batch=4, grad_accum=128, max_iters=5700, warmup=700, eval_every=500, eval_iters=40),
}


def get_batch(data, B, ctx, device):
    ix = torch.randint(len(data) - ctx - 1, (B,))
    x = torch.stack([torch.from_numpy(data[i:i + ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + ctx].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


def lr_mult(it, tc: TrainConfig):
    """Schedule as a fraction of peak lr (warmup -> flat -> linear cooldown)."""
    if it < tc.warmup:
        return (it + 1) / tc.warmup
    decay_start = int(tc.max_iters * (1 - tc.cooldown_frac))
    if it < decay_start:
        return 1.0
    frac = (it - decay_start) / max(1, tc.max_iters - decay_start)
    return tc.min_frac + (1 - tc.min_frac) * (1 - frac)


def make_optimizers(model, tc: TrainConfig):
    """AdamW baseline, or the Muon/AdamW hybrid (§5). Each param group carries a
    'base_lr' so the schedule can scale it by lr_mult()."""
    def tag(opt):
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
        return opt
    if tc.optim == "adamw":
        return [tag(torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=(0.9, 0.95), weight_decay=tc.wd))]
    from muon import Muon
    muon_p, adamw_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # embeddings + unembedding + norms -> AdamW; other 2-D hidden matrices -> Muon
        (adamw_p if ("wte" in n or "lm_head" in n or p.ndim < 2) else muon_p).append(p)
    return [tag(Muon(muon_p, lr=tc.muon_lr, momentum=0.95)),
            tag(torch.optim.AdamW(adamw_p, lr=tc.lr, betas=(0.9, 0.95), weight_decay=tc.wd))]


@torch.no_grad()
def sample_story(raw_model, tok, device, bos, n=100):
    raw_model.eval()
    idx = torch.tensor([[bos]], device=device)
    out = raw_model.generate(idx, n, temperature=0.8, top_k=40)[0].tolist()
    raw_model.train()
    return tok.decode(out[1:])  # drop the bos


@torch.no_grad()
def evaluate(model, data, tc, device, ctx):
    model.eval()
    losses = torch.zeros(tc.eval_iters)
    for k in range(tc.eval_iters):
        x, y = get_batch(data, tc.batch, ctx, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def run(tc: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc.seed)
    torch.set_float32_matmul_precision("high")
    outdir = ROOT / tc.out
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "ckpt.pt"

    cfg = PRESETS[tc.preset]
    tok = Tokenizer.from_file(str(ROOT / tc.tokenizer))
    bos = tok.token_to_id("<|bos|>")
    train_data = np.memmap(ROOT / tc.data / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(ROOT / tc.data / "val.bin", dtype=np.uint16, mode="r")
    raw_model = GPT(cfg).to(device)
    opts = make_optimizers(raw_model, tc)

    start_iter, best_val = 0, float("inf")
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ck["model"])
        for o, sd in zip(opts, ck["opts"]):
            o.load_state_dict(sd)
        start_iter, best_val = ck["iter"] + 1, ck.get("best_val", float("inf"))
        print(f"resumed from {ckpt_path} at iter {start_iter}")

    try:
        model = torch.compile(raw_model)   # ~2x on this GPU; keep raw_model for state_dict/generate
    except Exception as e:
        print("torch.compile off:", e); model = raw_model

    eff = tc.batch * tc.grad_accum * cfg.ctx
    print(f"{tc.preset}: {raw_model.num_params()/1e6:.1f}M params | eff batch {eff:,} tok/step "
          f"| train {len(train_data):,} tok | device {device}")

    csv_path = outdir / "log.csv"
    new_csv = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    writer = csv.writer(csv_f)
    if new_csv:
        writer.writerow(["iter", "train_loss", "val_loss", "lr", "elapsed_s"])

    t0 = last_ckpt = time.time()
    for it in range(start_iter, tc.max_iters):
        mult = lr_mult(it, tc)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * mult

        for o in opts:
            o.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(tc.grad_accum):
            x, y = get_batch(train_data, tc.batch, cfg.ctx, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / tc.grad_accum).backward()
            train_loss += loss.item() / tc.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        for o in opts:
            o.step()

        if it % 50 == 0:
            print(f"iter {it:5d}/{tc.max_iters} | loss {train_loss:.3f} | lrx {mult:.2f} | {time.time()-t0:.0f}s")

        if it % tc.eval_every == 0 or it == tc.max_iters - 1:
            val = evaluate(model, val_data, tc, device, cfg.ctx)
            cur_lr = opts[0].param_groups[0]["lr"]  # primary optimizer's scheduled lr
            writer.writerow([it, f"{train_loss:.4f}", f"{val:.4f}", f"{cur_lr:.2e}", f"{time.time()-t0:.0f}"])
            csv_f.flush()
            best_val = min(best_val, val)
            story = sample_story(raw_model, tok, device, bos)
            print(f"  eval iter {it}: val_loss {val:.4f} (best {best_val:.4f})")
            print(f"  sample: {story!r}")

        if time.time() - last_ckpt > tc.ckpt_secs or it == tc.max_iters - 1:
            torch.save({"model": raw_model.state_dict(), "opts": [o.state_dict() for o in opts], "iter": it,
                        "best_val": best_val, "cfg": asdict(tc), "preset": tc.preset}, ckpt_path)
            last_ckpt = time.time()
            print(f"  checkpoint @ iter {it} -> {ckpt_path}")

    csv_f.close()
    dt = time.time() - t0
    final_story = sample_story(raw_model, tok, device, bos)
    log_report(f"- **{tc.preset} [{tc.optim} seed{tc.seed}]** ({datetime.now():%Y-%m-%d %H:%M}) | "
               f"{raw_model.num_params()/1e6:.1f}M | {it+1} iters, {(it+1)*eff/1e6:.0f}M tok | "
               f"{dt/60:.1f} min | val_loss {best_val:.4f}\n  - sample: {final_story!r}")
    print(f"\ndone: {dt/60:.1f} min | best val_loss {best_val:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "run"])
    ap.add_argument("--preset", default="micro_30m")
    ap.add_argument("--optim", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max_iters", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--grad_accum", type=int)
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke()
    else:
        tc = TrainConfig(preset=a.preset, optim=a.optim)
        for k, v in RUN_DEFAULTS.get(a.preset, {}).items():
            setattr(tc, k, v)
        tc.out = a.out or f"runs/{a.preset}_{a.optim}"
        if a.seed is not None: tc.seed = a.seed
        if a.max_iters: tc.max_iters = a.max_iters
        if a.batch: tc.batch = a.batch
        if a.grad_accum: tc.grad_accum = a.grad_accum
        run(tc)
