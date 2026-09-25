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
- Time split: first 60% train, next 20% validation, last 20% test.
- Scaling: MinMax `[0, 1]` for German load and German weather variables, fitted on
  the train data only.
- Network: recurrent encoder-decoder with a one-layer `torch.nn.LSTM` core.
- Optimizer: Adam, learning rate `9.9027931032814e-05`, batch size 32.
- Maximum epochs: 50; early-stopping patience: 7.
- Training loss: Gaussian negative log likelihood (`NllGauss`).
- Seed: `torch.manual_seed(1)` from the original `src/train.py`.

## Expected evaluation protocol

Load `artifacts/paper_reproduction/opsd_recurrent.pkl` and evaluate only on the held-out
test partition. Report both point and probabilistic quality:

- RMSE and MAE;
- NLL / NllGauss;
- PICP (prediction interval coverage);
- sharpness and MIS.

For a fair comparison, use the same test period and 24-hour horizon for Persistence,
Seasonal Naive, Naive-STL, SARIMA, Prophet and the LSTM.
