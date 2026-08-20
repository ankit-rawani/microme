#!/bin/bash
# Architecture ablation lab (§9). Fixed protocol: micro_30m, TinyStories, Muon,
# 3000 iters, seed 0 — one arch change per run. ~48 min each.
D=/home/ankit/projects/my-mini-lm
C="run --preset micro_30m --optim muon --seed 0 --max_iters 3000"
echo "ablation lab start $(date)"

python "$D/train.py" $C --tag baseline               --out "$D/runs/ablate_baseline" > "$D/ablate_baseline.log" 2>&1
echo "baseline done (exit $?) $(date)"
python "$D/train.py" $C --mlp swiglu   --tag swiglu   --out "$D/runs/ablate_swiglu"   > "$D/ablate_swiglu.log"   2>&1
echo "swiglu done (exit $?) $(date)"
python "$D/train.py" $C --layer_reuse 2 --tag reuse2  --out "$D/runs/ablate_reuse2"   > "$D/ablate_reuse2.log"   2>&1
echo "reuse2 done (exit $?) $(date)"
python "$D/train.py" $C --tie           --tag tied    --out "$D/runs/ablate_tied"     > "$D/ablate_tied.log"     2>&1
echo "tied done (exit $?) $(date)"

echo "ABLATE COMPLETE $(date)"
