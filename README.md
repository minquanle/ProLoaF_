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

## Current status and next work

The LSTM training stage is complete. The next stage is evaluation of the saved
checkpoint on the held-out test period, followed by the paper baselines: Persistence,
Seasonal Naive, Naive-STL, SARIMA and Prophet. GRU is deliberately not treated as the
paper model; it may be added later as a separate extension with the same data split
and metrics.

See [`reproduction/REPRODUCTION_PROTOCOL.md`](reproduction/REPRODUCTION_PROTOCOL.md)
and [`reproduction/RUN_CARD.md`](reproduction/RUN_CARD.md) for the full provenance.
