# Completed DLinear OPSD experiment, 2026-10-15

Status: all three seeds trained and evaluated successfully. Total runner wall
time was 57.83 seconds, excluding source download/setup and independent checks.
The official DLinear model file was not changed. Source commit:
`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`.

## Results on C, original point forecasts

- Seed 1: selected epoch 16, stopped at 23; RMSE 3,549.48 MW, MAE 2,392.63 MW.
- Seed 2: selected epoch 18, stopped at 25; RMSE 3,586.94 MW, MAE 2,462.03 MW.
- Seed 3: selected epoch 12, stopped at 19; RMSE 3,573.58 MW, MAE 2,444.47 MW.
- Three-seed RMSE mean/sample std: 3,570.00 / 18.98 MW.
- Three-seed MAE mean/sample std: 2,433.05 / 36.08 MW.
- Reference strict-split LSTM seed 1: RMSE 3,652.55 MW, MAE 2,457.65 MW.
- Seasonal Naive 24: RMSE 6,932.13 MW, MAE 4,541.11 MW.
- Simple Naive: RMSE 11,120.70 MW, MAE 8,777.72 MW.

DLinear mean RMSE is about 2.26% lower and mean MAE about 1.00% lower than
the existing strict LSTM run. Do not select/report only the best C seed: all
three are retained, and seed 2's MAE is slightly higher than the LSTM's.
This is not evidence of universal superiority or statistical significance.

## Protocol and caveats

Same 43,824 hourly OPSD rows, A/B/C boundaries, training-only MinMax scale,
147-hour input length, 24-hour output length, 8,595 C origins and 206,280
forecast-target pairs as the strict LSTM run. Each model selects its checkpoint
on B, not on C. All three DLinear checkpoints were frozen before C inference.
Only historical DE load is supplied to DLinear. The reference LSTM additionally
uses weather, Austrian load and future calendar features, and a different loss.
Therefore this compares two configured systems, not equal-input architectures.
The C period was already inspected in previous project experiments.

DLinear has 7,104 parameters and no probabilistic output. No sigma, calibration,
PICP or MIS is reported. The learning schedule and loaders are the OPSD adapter's
choices, not a reproduction of the original DLinear paper benchmark. Read
`reproduction/DLINEAR_PROTOCOL.md` for the complete scope and commands.

## Verification and saved evidence

The read-only verification command passed for all three seeds:

```powershell
python .\reproduction\verify_dlinear_opsd.py --run-dir .\artifacts\paper_reproduction\dlinear_opsd_20261015
```

It checks source/data/checkpoint hashes, restored-model predictions, every target
timestamp, alignment with the strict LSTM target cache, and recomputed metrics.

Artifacts include `protocol.json`, `epoch_history.csv`, one checkpoint and
prediction archive per seed, `test_summary_per_seed.csv`,
`test_metrics_by_horizon.csv`, `comparison_point_metrics.csv`,
`aggregate_three_seeds.json`, and `status.json`. Checkpoints store state_dict,
seed, best epoch and protocol. Raw code/data and previous model artifacts remain
unchanged. The report, slides, demo and GitHub have not been updated by this run.
