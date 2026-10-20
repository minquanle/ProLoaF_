# Artifacts from the first local reproduction run

- `opsd_recurrent.pkl`: best LSTM checkpoint selected by validation early stopping.
- `training.log`: complete console log for the run.
- `tensorboard/`: TensorBoard event file with epoch-level training data.
- `training_config_used.json`: exact OPSD configuration used for the run.
- `evaluation/`: original LSTM metric table, plots, and compact forecast tensors.
- `baseline_comparison/`: Simple Naive and Seasonal Naive results.
- `prophet_paper_protocol/`: Prophet notebook-protocol reproduction results.
- `interval_calibration/`: validation-only interval calibration and frozen test results.
- `strict_602020_20261013/`: new checkpoint selected on B only, final C results,
  complete epoch history and comparison with the old checkpoint. No calibration.
- `residual_comparison/`: author-method residual histogram and provenance;
  this still uses the old checkpoint, not the strict-split model.

The first checkpoint used A for gradients and B+C for early stopping. Its C
results and calibration are not an independent test evaluation. Only the new
strict-split run excluded C from checkpoint selection. Reports/slides/demo have
not been automatically switched to the new model.

Submission reports and presentation slides are intentionally excluded from Git.

The separate `dlinear_opsd_20261015/` extension applies the official DLinear
architecture to OPSD. It includes three seed checkpoints, predictions, point
metrics and comparison to the strict LSTM. This is a load-only model without
prediction intervals, not an equal-input architectural benchmark or a full
reproduction of the DLinear paper's published experiment.

The strict LSTM was also retrained with seeds 2 and 3
(`strict_602020_seed2/`, `strict_602020_seed3/`, created with
`run_lstm_strict_602020.py --seed N`). `strict_602020_3seeds_summary/` holds the
per-seed table, mean +/- sample std and RMSE by horizon averaged over the seeds.

The artifacts were produced from the paper source at commit
`1055d1fd3ae00ffce0b67eed03ec84a7410f5da8` on 2026-09-24. Their hashes are recorded
in `reproduction/RUN_CARD.md`.
