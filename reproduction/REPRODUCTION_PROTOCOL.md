# Reproduction protocol

## Scope

This is a reproduction of the OPSD example supplied with the ProLoaF paper source.
It is not a newly designed GRU project. Any future GRU result must be reported as an
extension, separate from this reproduction.

## Frozen upstream source

- Repository: `https://github.com/ElsevierSoftwareX/SOFTX-D-21-00190`
- Pinned commit: `1055d1fd3ae00ffce0b67eed03ec84a7410f5da8`
- Local location after clone: `paper_source/`

The upstream source includes its own data submodule. Clone this repository with
`--recurse-submodules` or run:

```powershell
git submodule update --init --recursive
```

## Experiment configuration

- Frequency: hourly.
- Prepared dataset: `paper_source/data/opsd.csv`.
- Target: German actual electricity load, `DE_load_actual_entsoe_transparency`.
- History horizon: 147 hours.
- Forecast horizon: 24 hours.
- Chronological regions: A=first 60%, B=next 20%, C=last 20%.
  The original train.py uses A for gradients and B+C for early stopping;
  evaluate.py then scores C. Thus the original C result is not independent.
  The new run_lstm_strict_602020.py uses only B for early stopping and holds C
  out until after checkpoint selection.
- Scaling: MinMax `[0, 1]` for German load and German weather variables, fitted on
  the train data only.
- Network: recurrent encoder-decoder with a one-layer `torch.nn.LSTM` core.
- Optimizer: Adam, learning rate `9.9027931032814e-05`, batch size 32.
- Maximum epochs: 50; early-stopping patience: 7.
- Training loss: Gaussian negative log likelihood (`NllGauss`).
- Seed: `torch.manual_seed(1)` from the original `src/train.py`.

## Expected evaluation protocol

The original checkpoint is `artifacts/paper_reproduction/opsd_recurrent.pkl`.
The independently evaluated checkpoint is
`artifacts/paper_reproduction/strict_602020_20261013/opsd_recurrent.pkl`.
Keep these separate: old calibration/demo refer to the former, not the latter.
Both score the same C windows, but only the new run excludes C from selection.
Report both point and probabilistic quality:

- RMSE and MAE;
- NLL / NllGauss;
- PICP (prediction interval coverage);
- sharpness and MIS.

For a fair comparison, use the same test period and 24-hour horizon for Persistence,
Seasonal Naive, Naive-STL, SARIMA, Prophet and the LSTM.
