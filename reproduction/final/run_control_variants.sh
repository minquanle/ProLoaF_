#!/usr/bin/env bash
# Train the two control variants (hol, noat) for seeds 1-5: ten runs in parallel, 1 CPU thread each.
# Usage (from repository root): bash reproduction/final/run_control_variants.sh /path/to/python
PY=${1:-python}
LOGS=artifacts/final_exam/logs
mkdir -p "$LOGS"
for variant in hol noat; do
  for seed in 1 2 3 4 5; do
    "$PY" reproduction/final/run_lstm_variant.py --variant "$variant" --seed "$seed" --threads 1       > "$LOGS/${variant}_seed${seed}.log" 2>&1 &
  done
done
wait
echo ALL_DONE
