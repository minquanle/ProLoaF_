# Reproduction artifacts

Every folder here is the committed output of one runner in `reproduction/`.
Test results refer to part C, the final 20% of the data (2019): 8,595 forecast
origins × 24 hours.

## Original-code LSTM (reference)

This checkpoint was trained on A, but early stopping used B and C together.
Its C results are therefore not an independent test evaluation.

| Path | Content |
|---|---|
| `opsd_recurrent.pkl` | Checkpoint selected at epoch 34 (stopped at epoch 41) |
| `training.log`, `tensorboard/` | Console log and epoch-level TensorBoard events |
| `training_config_used.json` | Exact OPSD configuration of the run |
| `evaluation/` | Upstream metric table, plots and compact forecast tensors |
| `interval_calibration/` | c = 0.841887 chosen on B, coefficient sweep, frozen C results |
| `residual_comparison/` | Redrawn Fig. 3 of the paper, with provenance notes |
| `prophet_paper_protocol/` | Prophet as in the authors' notebook, compared with this checkpoint |

## Strict 60/20/20 LSTM (main result)

In these runs, A fits the weights and the scaler, and only B selects the
checkpoint. C is evaluated after the checkpoint is frozen. The architecture
and hyperparameters are unchanged; only the seed differs between runs.

| Path | Seed | Best / stopped epoch | RMSE (MW) | MAE (MW) |
|---|---:|---|---:|---:|
| `strict_602020_20261013/` | 1 | 15 / 22 | 3,652.55 | 2,457.65 |
| `strict_602020_seed2/` | 2 | 44 / 50 | 3,259.89 | 2,075.71 |
| `strict_602020_seed3/` | 3 | 30 / 37 | 3,151.75 | 2,085.38 |
| `strict_602020_3seeds_summary/` | mean ± sample std | | 3,354.73 ± 263.53 | 2,206.25 ± 217.77 |

Each run folder has the same files:

- `protocol.json`: partition boundaries, hashes, seed and thread count;
- `status.json`: progress record; results are valid only when it says `complete`;
- `epoch_history.csv`: training loss on A and validation loss on B for every epoch;
- `opsd_recurrent.pkl`: selected checkpoint, including scaler and configuration;
- `test_predictions.npz`: all C predictions with μ and σ;
- `test_summary.csv`, `test_metrics_by_horizon.csv`: C metrics overall and per horizon;
- `paper_metrics.csv`: values from the upstream metric definitions, for comparison;
- `comparison_old_vs_strict.csv`: comparison with the original-code checkpoint on the same windows.

Protocol details are in [`strict_602020_20261013/README.md`](strict_602020_20261013/README.md).
`strict_602020_seed2.log` and `strict_602020_seed3.log` are the console logs of
seeds 2 and 3. `strict_602020_3seeds_summary/` is written by
`reproduction/aggregate_lstm_seeds.py`.

No interval calibration has been applied to the strict runs.

## Baselines and extensions

| Path | Content |
|---|---|
| `baseline_comparison/` | Simple Naive and Seasonal Naive, with intervals taken from B residual quantiles |
| `dlinear_opsd_20261015/` | DLinear, three seeds, using German load history only; point forecasts without intervals |

## Provenance

All runs use the paper source at commit
`1055d1fd3ae00ffce0b67eed03ec84a7410f5da8`. The original-code run finished on
2026-09-24. File hashes are listed in `reproduction/RUN_CARD.md`.

The submission report and slides are intentionally excluded from Git.
