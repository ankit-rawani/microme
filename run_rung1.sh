#!/bin/bash
# Ablation rung 1 (AdamW vs Muon), quick single-seed: 3000 iters each (~50 min/run).
# Results auto-append to REPORT.md (labeled [adamw seed0] / [muon seed0]).
# NOT set -e: if one run dies, still attempt the other.
D=/home/ankit/projects/my-mini-lm
echo "rung 1 start $(date)"
python "$D/train.py" run --optim adamw --seed 0 --max_iters 3000 > "$D/rung1_adamw.log" 2>&1
echo "adamw done (exit $?) $(date)"
python "$D/train.py" run --optim muon  --seed 0 --max_iters 3000 > "$D/rung1_muon.log" 2>&1
echo "muon done (exit $?) $(date)"
echo "RUNG1 COMPLETE $(date) — see REPORT.md"
