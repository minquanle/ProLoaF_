# Run card: first local LSTM reproduction

## Identity

- Run date: 2026-09-24.
- Command: `python .\src\train.py -s opsd` from `paper_source/`.
- Upstream commit: `1055d1fd3ae00ffce0b67eed03ec84a7410f5da8`.
- Model name: `opsd_recurrent`.
- Model class: probabilistic encoder-decoder LSTM.
- Runtime: Windows CPU; CUDA was unavailable.

## Software

- Python 3.9.25
- PyTorch 1.13.1+cpu
- pandas 1.5.3
- pmdarima 1.8.5
- Prophet 1.1.5

## Data and tensors

- Prepared dataset: 43,824 hourly rows and 55 columns, 2015-01-01 to 2019-12-31.
- Configured boundaries: 60% and 80%. Actual original train.py behavior:
  A=60% for gradients, B+C=40% for validation/early stopping. evaluate.py
  then scores C=last 20%, which is not independent of checkpoint selection.
- Encoder tensor per sample: 147 historical hours x 5 features.
- Decoder tensor per sample: 24 future hours x 46 calendar features.
- Target tensor per sample: 24 future hours x 1 German-load value.

## Optimisation result

- Maximum epochs: 50.
- Actual epochs run: 41.
- Best observed validation loss: approximately `-2.35` at epoch 34.
- Stop condition: 7 epochs without validation improvement.
- Training wall time: 1 hour 11 minutes 47 seconds.
- The final checkpoint reloads the best validation weights, not epoch-41 weights.

## Artifact checksums (SHA-256)

```text
opsd_recurrent.pkl
C1F3446055D34204C6A5F1EE285939108B1D0308E4F1323F85F5FEF5350AFBC7

training.log
91973DC0BF97A4408848DC40F6B86E22E8146D75D3C925238BF8BDE4B30C9737

tensorboard/events.out.tfevents.1790235848.DESKTOP-A0A9J2U.9180.0
E95B9EACFBE63BF9FFA9A9DA11F85AB9E3A29726269E6B84E5543914A19D3834
```

## Subsequent completed stages

Original C evaluation, two Naive baselines, aligned Prophet comparison,
validation-selected interval calibration and the local demo are complete.
The original checkpoint's selection used B+C, including the final C region.

A separate strict 60/20/20 run completed on 2026-10-13. It stopped at epoch 22,
selected epoch 15 on B alone, and scored C after selection: RMSE 3,652.55 MW,
MAE 2,457.65 MW, PICP 97.1015%, original intervals with c=1. See
`artifacts/paper_reproduction/strict_602020_20261013/` for exact evidence.
The demo and original calibration still use the old checkpoint. No GRU result
or calibration of the new checkpoint is claimed.
