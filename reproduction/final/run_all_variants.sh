#!/usr/bin/env bash
# Train both final-exam variants for seeds 1-5: five runs in parallel, 3 CPU threads each.
# Usage (from repository root): bash reproduction/final/run_all_variants.sh /path/to/python
PY=${1:-python}
LOGS=artifacts/final_exam/logs
mkdir -p "$LOGS"
for variant in at at_hol; do
  for seed in 1 2 3 4 5; do
    "$PY" reproduction/final/run_lstm_variant.py --variant "$variant" --seed "$seed" --threads 3 \
      > "$LOGS/${variant}_seed${seed}.log" 2>&1 &
  done
  wait
done
echo ALL_DONE
