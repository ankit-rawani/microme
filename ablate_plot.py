"""Overlay val-loss curves + sample-efficiency table for the architecture lab."""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RUNS = {"baseline": "ablate_baseline", "swiglu": "ablate_swiglu",
        "layer_reuse=2": "ablate_reuse2", "tied embeds": "ablate_tied"}


def load(stem):
    it, v = [], []
    p = ROOT / "runs" / stem / "log.csv"
    if not p.exists():
        return None, None
    for r in csv.DictReader(open(p)):
        it.append(int(r["iter"])); v.append(float(r["val_loss"]))
    return np.array(it), np.array(v)


def iters_to(it, v, tgt):
    for k in range(1, len(v)):
        if v[k] <= tgt:
            f = (v[k - 1] - tgt) / (v[k - 1] - v[k]); return it[k - 1] + f * (it[k] - it[k - 1])
    return None


plt.figure(figsize=(7.5, 4.8))
base_it, base_v = load(RUNS["baseline"])
print(f"{'variant':16s} {'final val':>10s} {'vs baseline':>12s}")
for name, stem in RUNS.items():
    it, v = load(stem)
    if it is None:
        print(f"{name:16s} {'(missing)':>10s}"); continue
    plt.plot(it, v, "-o", ms=2.5, label=f"{name} ({v[-1]:.3f})")
    delta = "" if base_v is None else f"{v[-1]-base_v[-1]:+.4f}"
    print(f"{name:16s} {v[-1]:>10.4f} {delta:>12s}")
plt.xlabel("iteration (65,536 tok/iter)"); plt.ylabel("val loss")
plt.title("Architecture ablation lab — micro_30m, TinyStories, Muon, 3000 it")
plt.ylim(1.3, 2.6); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
out = ROOT / "runs" / "ablate_arch.png"
plt.savefig(out, dpi=120)
print("saved", out)
