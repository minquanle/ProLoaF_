# DLinear extension on OPSD

## What this experiment is

Apply DLinear from *Are Transformers Effective for Time Series Forecasting?*
(Zeng et al., AAAI 2023; DOI 10.1609/aaai.v37i9.26317) to the existing German
load task. This is not a reproduction of the original paper's datasets,
forecast lengths, training schedule, or published benchmark scores.

The architecture is loaded unmodified from the official repository:
https://github.com/cure-lab/LTSF-Linear

Pinned commit: `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`.
Source: `models/DLinear.py`; Apache-2.0 license in the source checkout.
The checkout is intentionally ignored by the parent Git repository. Setup
fetches the pinned source; no upstream requirements installation is necessary
because the imported model needs only the existing PyTorch/NumPy environment.

## Run

From the project root:

```powershell
conda activate proloaf-paper-repro
powershell -ExecutionPolicy Bypass -File .\reproduction\setup_dlinear_source.ps1
python .\reproduction\run_dlinear_opsd.py --output-dir .\artifacts\paper_reproduction\dlinear_new_run
```

Use a new output directory. Add `--dry-run` to check source identity, partition
sizes, tensor shapes, decomposition reconstruction, and parameter count without
training. This is a small smoke check, not proof of benchmark correctness.

## Fixed protocol, chosen before C evaluation

- Dataset: exact author-prepared `paper_source/data/opsd.csv`, 43,824 hours.
- Sole input and target: historical German load. No Austrian load, weather,
  future calendar features, holiday labels, or actual future load as input.
- Train A: first 26,294 hours. Validation B: next 8,765. Test C: last 8,765.
- MinMax target scaling uses only A (minimum 31,307 MW, range 46,242 MW).
- Each sample is 147 past hours and 24 future hours. Construct windows
  independently in A/B/C, as in the existing LSTM evaluation.
- Train: 26,124 windows. B/C: 8,595 windows each. No windows cross partitions.
- Official architecture: kernel 25, individual=False, one input channel,
  7,104 trainable parameters.
- Adapter training: MSE, Adam, fixed learning rate 0.001, batch 32, up to
  50 epochs, patience 7, CPU with four threads. These are the OPSD experiment's
  training choices, not a claim to preserve the full original benchmark loop.
- Shuffle training windows only. Validation/test are ordered and retain every
  window. Train also retains the final partial batch.
- Seeds 1, 2, 3 are all reported. Select best checkpoint per seed by B MSE only.
  Freeze all three checkpoints before constructing C prediction tensors.
- No calibration or intervals: DLinear predicts only 24 point values.

## Comparison and limitations

All C predictions are aligned and checked against cached target tensors of the
strict-split LSTM run, giving 206,280 forecast-target pairs. Recompute all point
metrics under one implementation and report MW. This comparison includes the
strict LSTM and two naive baselines. Do not select the best seed by C results;
report all seeds and the mean/sample standard deviation.

The LSTM has five historical covariates plus future calendar features and
Gaussian NLL training, whereas DLinear has only load and MSE training. Thus
this compares two configured forecasting systems, not architecture effects
under equal input information. An additional load-only LSTM would be needed
for the latter, and has not been run here. The old LSTM selected using B+C is
not used as the main comparison. Three DLinear seeds versus one LSTM seed do
not establish a statistically significant ranking.

C has already been inspected in earlier project experiments. Its labels are
excluded from the new model fitting/selection, but it is not a fresh unseen
dataset for the overall research process. Do not claim otherwise or tune more
models against the C result without disclosing repeated use.

Training timing is reported for DLinear only. Prior LSTM training/evaluation
used different implementations and conditions; wall times are not a controlled
speed benchmark. No PICP or MIS is invented for a model without intervals.

The report, presentation, demo, old checkpoints and GitHub are not automatically
updated by this runner. Results are saved in its explicitly chosen directory.
