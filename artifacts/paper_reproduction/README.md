# Artifacts from the first local reproduction run

- `opsd_recurrent.pkl`: best LSTM checkpoint selected by validation early stopping.
- `training.log`: complete console log for the run.
- `tensorboard/`: TensorBoard event file with epoch-level training data.
- `training_config_used.json`: exact OPSD configuration used for the run.
- `evaluation/`: original LSTM metric table, plots, and compact forecast tensors.
- `baseline_comparison/`: Simple Naive and Seasonal Naive results.
- `prophet_paper_protocol/`: Prophet notebook-protocol reproduction results.
- `interval_calibration/`: validation-only interval calibration and frozen test results.

Submission reports and presentation slides are intentionally excluded from Git.

The artifacts were produced from the paper source at commit
`1055d1fd3ae00ffce0b67eed03ec84a7410f5da8` on 2026-09-24. Their hashes are recorded
in `reproduction/RUN_CARD.md`.
