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
checkpoint used its configured 60/20/20 split and training-only scaling.

## Post-hoc interval calibration extension

The paper-reproduction LSTM outputs both a point forecast (the Gaussian mean
`mu`) and an uncertainty estimate (`sigma`). The original 95% interval on the
held-out test data had empirical coverage of 97.10%, so it was slightly too wide.
This repository adds a clearly separated post-hoc extension: standard-deviation
scaling. It does **not** retrain or alter the LSTM point forecast.

First choose a multiplier only on the validation period (the original OPSD rows
60% to 80%):

```powershell
conda activate proloaf-paper-repro
python .\reproduction\calibrate_lstm_intervals.py
```

Then evaluate that frozen multiplier exactly as selected on the held-out test
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

## Current status and next work

Completed: LSTM training and test evaluation, Simple Naive and Seasonal Naive
baselines, the Prophet notebook reproduction, and the validation-only interval
calibration extension. The next deliverables are an interactive demonstration and
presentation slides. GRU is deliberately not treated as the paper model; it may be
added later as a separate extension with the same data split and metrics.

See [`reproduction/REPRODUCTION_PROTOCOL.md`](reproduction/REPRODUCTION_PROTOCOL.md)
and [`reproduction/RUN_CARD.md`](reproduction/RUN_CARD.md) for the full provenance.
