"""3-seed significance check for the arch ablations: mean ± std + noise verdict."""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
VARIANTS = {"baseline": "ablate_baseline", "tied embeds": "ablate_tied", "layer_reuse=2": "ablate_reuse2"}


def final_val(dirname):
    p = ROOT / "runs" / dirname / "log.csv"
    if not p.exists():
        return None
    return float(list(csv.DictReader(open(p)))[-1]["val_loss"])


res = {}
for name, stem in VARIANTS.items():
    vals = [final_val(stem)] + [final_val(f"{stem}_s{s}") for s in (1, 2)]   # s0 has no suffix
    res[name] = np.array([v for v in vals if v is not None])
    print(f"{name:16s} seeds {[f'{v:.4f}' for v in res[name]]}  mean {res[name].mean():.4f}  std {res[name].std(ddof=1):.4f}")

print()
base = res["baseline"]
for name in ("tied embeds", "layer_reuse=2"):
    v = res[name]
    dmean = v.mean() - base.mean()
    pooled = np.sqrt(v.std(ddof=1) ** 2 + base.std(ddof=1) ** 2)
    verdict = "SURVIVES noise" if abs(dmean) > pooled else "within noise (not significant)"
    print(f"{name:16s} vs baseline: Δmean {dmean:+.4f}, pooled σ {pooled:.4f}  ->  {verdict}")
