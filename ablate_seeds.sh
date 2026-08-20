#!/bin/bash
# 3-seed confirmation of the interesting winners (§9). Seed 0 already exists in
# runs/ablate_{baseline,tied,reuse2}; this adds seeds 1 and 2.
D=/home/ankit/projects/my-mini-lm
C="run --preset micro_30m --optim muon --max_iters 3000"
for s in 1 2; do
  python "$D/train.py" $C --seed $s               --tag baseline --out "$D/runs/ablate_baseline_s$s" > "$D/ablate_baseline_s$s.log" 2>&1
  echo "baseline s$s done (exit $?) $(date)"
  python "$D/train.py" $C --seed $s --tie          --tag tied     --out "$D/runs/ablate_tied_s$s"     > "$D/ablate_tied_s$s.log"     2>&1
  echo "tied s$s done (exit $?) $(date)"
  python "$D/train.py" $C --seed $s --layer_reuse 2 --tag reuse2   --out "$D/runs/ablate_reuse2_s$s"   > "$D/ablate_reuse2_s$s.log"   2>&1
  echo "reuse2 s$s done (exit $?) $(date)"
done
echo "SEEDS COMPLETE $(date)"
