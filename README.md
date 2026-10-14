# ProLoaF paper reproduction

This repository records a local reproduction of the OPSD illustrative experiment from
**“ProLoaF: Probabilistic load forecasting for power systems”**. The goal is to run the
paper's code and configuration before adding any independent extension.

## What has been reproduced

- Official paper source is pinned as the `paper_source` Git submodule at commit
  `1055d1fd3ae00ffce0b67eed03ec84a7410f5da8`.
- Dataset: the paper's prepared `data/opsd.csv` from its nested data submodule,
  with 43,824 hourly observations from 2015-01-01 through 2019-12-31.
- Target: `DE_load_actual_entsoe_transparency`.
- Model: one-layer probabilistic encoder-decoder LSTM.
- Input/output: 147 hours of history -> 24 hours ahead.
- Training finished on 2026-09-24 with early stopping at epoch 41; the best weights
  were saved from epoch 34.

The checkpoint, training log and TensorBoard event file are versioned in
[`artifacts/paper_reproduction/`](artifacts/paper_reproduction/).

## Repository layout

```text
ProLoaF_/
├── paper_source/                 # official source, pinned as a Git submodule
├── artifacts/paper_reproduction/ # checkpoint and training evidence from this run
├── reproduction/                 # protocol and exact run record
├── data/README.md                # data provenance; raw files are not duplicated here
├── notebooks/README.md           # points to original notebooks in paper_source
└── requirements.txt              # pinned Python dependencies for this reproduction
```

## Clone and set up

Clone recursively so that both the paper source and its original data submodule are
available:

```powershell
git clone --recurse-submodules https://github.com/minquanle/ProLoaF_.git
cd ProLoaF_
conda create -n proloaf-paper-repro python=3.9 -y
conda activate proloaf-paper-repro
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-deps -e .\paper_source
python -m ipykernel install --user --name proloaf-paper-repro --display-name "Python (ProLoaF Paper Reproduction)"
```

## Re-run training

```powershell
conda activate proloaf-paper-repro
cd paper_source
python .\src\train.py -s opsd
```

The original code sets `torch.manual_seed(1)`. It uses the default Gaussian negative
log-likelihood loss (`NllGauss`) and early stopping with patience 7.

## Prophet benchmark from the paper notebook

The executable implementation of the Prophet section in the original
`Train and Benchmark Models.ipynb` is
[`reproduction/run_prophet_paper_protocol.py`](reproduction/run_prophet_paper_protocol.py).
It fits **one** Prophet model on the first 80% of OPSD and produces a continuous
hourly forecast, as in the notebook. Run it from the repository root:

```powershell
conda activate proloaf-paper-repro
python .\reproduction\run_prophet_paper_protocol.py
```

Outputs are written to `artifacts/paper_reproduction/prophet_paper_protocol/`.
The runner exports two labels on purpose:

- `paper_literal_80pct`: literal notebook behavior, including Prophet's default
  80% interval and its 147-hour timestamp offset. It is source-fidelity evidence,
  not a valid LSTM comparison.
- `chronological_aligned_95pct`: the same single fitted Prophet model, with forecast
  and actual timestamps aligned and a 95% interval. This is the result appropriate
  for an indicative comparison to the LSTM.

The comparison remains indicative rather than strictly fair: Prophet's notebook
uses an 80/20 fit/test split and full-data target scaling, while the existing LSTM
checkpoint used 60% for gradients and the remaining 40% (B+C) for checkpoint
selection, with training-only scaling. Its final C evaluation is not independent
of early stopping.

## Post-hoc interval calibration extension

The paper-reproduction LSTM outputs both a point forecast (the Gaussian mean
`mu`) and an uncertainty estimate (`sigma`). The original 95% interval on the
final C evaluation had empirical coverage of 97.10%, so it was slightly too wide.
This repository adds a clearly separated post-hoc extension: standard-deviation
scaling. It does **not** retrain or alter the LSTM point forecast.

First choose a multiplier only on the validation period (the original OPSD rows
60% to 80%):

```powershell
conda activate proloaf-paper-repro
python .\reproduction\calibrate_lstm_intervals.py
```

Then evaluate that frozen multiplier exactly as selected on the final C
period (80% to 100%):

```powershell
python .\reproduction\evaluate_calibrated_lstm_on_test.py
```

For this run, validation selected `c = 0.841887`. On test, PICP changed from
97.10% to **95.06%**, while Sharpness decreased from 0.268506 to **0.226052**.
RMSE and MAE stay unchanged because calibration only rescales the interval around
the same point forecast. Results, the full coefficient sweep and the validation
trade-off figure are in
[`artifacts/paper_reproduction/interval_calibration/`](artifacts/paper_reproduction/interval_calibration/).

`MIS_standard_95` is the standard alpha=0.05 mean interval score. It is reported
separately from `ProLoaF_legacy_MIS`: the latter preserves the value printed by
the original source, whose `Mis.from_quantiles` method includes the median when
selecting alpha and therefore does not score the displayed 95% interval.

Calibration only selects c on B, but the old checkpoint was already selected
using B+C. This does not retroactively make C an independent test set.

## Strict 60/20/20 experiment (2026-10-13)

The separate runner fixes the split without editing the upstream submodule or
overwriting the old checkpoint. A contains 26,294 hours, B 8,765, and C 8,765.
Only A fits the scaler and updates weights; only B selects the checkpoint.
The selected checkpoint is frozen before C evaluation. Run from the repository root:

```powershell
conda activate proloaf-paper-repro
python .\reproduction\run_lstm_strict_602020.py --output-dir .\artifacts\paper_reproduction\strict_602020_new_run --threads 4
```

The output directory must be new. The completed run is in
[`strict_602020_20261013`](artifacts/paper_reproduction/strict_602020_20261013/).
Training stopped at epoch 22 and selected epoch 15. Including evaluation, the run
took about 27 minutes 33 seconds. On the same 8,595 C windows, original intervals
(c=1) give RMSE 3,652.55 MW, MAE 2,457.65 MW, PICP 97.1015%, and mean interval
width 15,927.92 MW. The old original-interval result had RMSE 3,262.64 MW and
MAE 2,117.27 MW. Fixing test independence does not guarantee lower error.

The architecture, seed, optimizer and training hyperparameters are preserved.
Four CPU threads are explicitly used; the old thread count is unknown, so the
difference cannot be attributed solely to the split from one run per protocol.
The old c=0.841887 has not been applied to the new checkpoint. The demo continues
to use the old checkpoint and calibration, not the new experiment.

## Auditable residual histogram comparison

```powershell
python .\reproduction\plot_residual_comparison.py
```

This uses the old checkpoint and saved aligned Prophet forecasts. Following the
author notebook, each origin contributes the signed mean residual of its 24
hours: 8,595 values per model, with 80 automatically selected bins per model.
The author-style view retains all data but zooms the horizontal axis; 169 LSTM
and 165 Prophet residuals are outside the visible frame. A full-range figure is
also saved. Source figure credit and limitations are recorded in
[`residual_comparison/README.md`](artifacts/paper_reproduction/residual_comparison/README.md).
Matching plot style does not establish reproduction of the exact published data.

## Interactive local demo

The local HTML dashboard lets you select one of the 8,595 test forecast origins. It
shows the 147 hours of load history, the next 24 actual hours, the LSTM point
forecast, and both the original and calibrated 95% intervals. It runs inference
only; it never retrains the checkpoint.

```powershell
conda activate proloaf-paper-repro
python .\demo\run_demo.py
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) in a browser. On the first
page load, the demo prepares all test forecasts in memory; this takes roughly
10 seconds. Later slider changes are immediate. The interface is built with
HTML, CSS and JavaScript; the local Python server only exposes model forecasts
through a small API.

## Current status

Completed: LSTM training and test evaluation, Simple Naive and Seasonal Naive
baselines, the Prophet notebook reproduction, the validation-only interval
calibration extension, and the interactive HTML demonstration. GRU is deliberately
not treated as the paper model; it may be added later as a separate extension with
the same data split and metrics.

Submission reports and presentation slides are kept as local deliverables and are
intentionally excluded from this source repository.

See [`reproduction/REPRODUCTION_PROTOCOL.md`](reproduction/REPRODUCTION_PROTOCOL.md)
and [`reproduction/RUN_CARD.md`](reproduction/RUN_CARD.md) for the full provenance.
