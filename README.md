# ProLoaF paper reproduction

This repository reproduces the OPSD example from **"ProLoaF: Probabilistic load
forecasting for power systems"** (Gürses-Tran, Oppermann and Monti, SoftwareX 23,
101487, 2023). It then extends that example.

**Task.** The model reads the previous 147 hours and forecasts the German
electricity load for each of the next 24 hours. Each forecast hour gets:

- a point forecast μ,
- a standard deviation σ,
- a nominal 95% interval μ ± 1.96σ.

The work has four parts:

1. **Reproduction.** Train the paper's encoder–decoder LSTM with the authors'
   code, configuration and data.
2. **Strict re-evaluation.** The upstream training code uses the final 2019
   test year for checkpoint selection. A corrected 60/20/20 run removes that
   overlap. It was trained with three seeds.
3. **Comparisons.** Simple Naive, Seasonal Naive, Prophet (following the
   authors' notebook) and DLinear (Zeng et al., AAAI 2023) on the same test
   year.
4. **Interval calibration and demo.** A single σ multiplier is chosen on the
   validation year. A local HTML dashboard shows the forecasts.

## Main results

All rows are scored on the same final 20% of the data (2019, part C):
8,595 forecast origins × 24 hours = 206,280 forecast–actual pairs. Values are
in MW, except PICP in %. "Width" is the mean width of the 95% interval.

| Method | RMSE | MAE | PICP | Width | MIS |
|---|---:|---:|---:|---:|---:|
| **Not tuned on 2019** | | | | | |
| Simple Naive | 11,120.70 | 8,777.72 | 95.82 | 38,565.64 | 41,235.87 |
| Seasonal Naive (24 h) | 6,932.13 | 4,541.11 | 94.92 | 31,732.90 | 35,700.18 |
| Prophet (notebook protocol, aligned) | 3,946.47 | 2,898.35 | 95.40 | 15,639.61 | 22,212.29 |
| **Strict LSTM, mean of seeds 1–3** | **3,354.73** | **2,206.25** | 97.23 | 14,010.54 | 17,734.96 |
| ± sample std across seeds | ± 263.53 | ± 217.77 | ± 0.93 | ± 1,971.19 | ± 1,375.47 |
| DLinear, mean of seeds 1–3 | 3,570.00 | 2,433.05 | – | – | – |
| ± sample std across seeds | ± 18.98 | ± 36.08 | | | |
| **2019 used for checkpoint selection (reference only)** | | | | | |
| Original-code LSTM | 3,262.64 | 2,117.27 | 97.10 | 12,416.27 | 16,395.35 |
| Original-code LSTM + calibration (c = 0.841887) | 3,262.64 | 2,117.27 | 95.06 | 10,453.10 | 16,308.16 |

How to read the table:

- **LSTM vs. Prophet and the naive baselines.** On average the strict LSTM has
  about 52% lower RMSE than Seasonal Naive and about 15% lower than Prophet.
- **LSTM vs. DLinear.** The LSTM's mean RMSE is about 6% lower than DLinear's.
  - That gap (about 215 MW) is smaller than the LSTM's spread across seeds.
  - Seed 1 alone is worse than every DLinear seed.
  - DLinear is more accurate for horizons 1–6, and the LSTM for horizons 7–24.
  - The LSTM's error at horizon 1 is unusually high in all three seeds.
- **Inputs differ.** The LSTM also sees Austrian load, weather and calendar
  features. DLinear sees only German load history. The comparison therefore
  does not isolate architecture quality.
- **Prophet is indicative.** It is fitted once on the first 80% and forecasts
  2019 in one continuous run, without hourly updates.
- **Calibration.** It narrows the interval by 15.81% and brings coverage to
  95.06%, without changing the point forecasts. It was applied to the
  original-code checkpoint, so it illustrates the method. It is not an
  independent test result.

## Repository layout

```text
ProLoaF_/
├── paper_source/                  # official ProLoaF source, Git submodule pinned at 1055d1f
├── reproduction/                  # runners, verifiers and protocol notes
├── artifacts/paper_reproduction/  # checkpoints, predictions, metrics and logs of every run
├── demo/                          # local HTML dashboard (server.py + static/)
├── data/README.md                 # data provenance; the dataset comes from the submodule
├── notebooks/README.md            # points to the original notebooks in paper_source
└── requirements.txt               # pinned Python environment (Windows, CPU)
```

| Script in `reproduction/` | Purpose |
|---|---|
| `run_lstm_strict_602020.py` | Strict 60/20/20 LSTM run (`--seed`, `--threads`, `--dry-run`) |
| `aggregate_lstm_seeds.py` | Per-seed table, mean ± std and RMSE by horizon for the three strict seeds |
| `run_two_baselines.py` | Simple and Seasonal Naive with validation-based intervals |
| `run_prophet_paper_protocol.py` | Prophet as in the authors' benchmark notebook |
| `calibrate_lstm_intervals.py` | Choose the σ multiplier c on the validation year only |
| `evaluate_calibrated_lstm_on_test.py` | Apply the frozen c to the test year |
| `plot_residual_comparison.py` | Redraw Fig. 3 of the paper (residual histogram) |
| `setup_dlinear_source.ps1` | Fetch the official DLinear source at the pinned commit |
| `run_dlinear_opsd.py` / `verify_dlinear_opsd.py` | Train and independently verify DLinear |

## Data and split

The dataset is the authors' prepared table `paper_source/data/opsd.csv`. It has
43,824 hourly rows covering 2015-01-01 to 2019-12-31, with no missing values.
See [`data/README.md`](data/README.md).

The data is split in chronological order:

| Part | Share | Hours | Windows (147 + 24) | Role |
|---|---|---:|---:|---|
| A | 0–60% | 26,294 | 26,124 | Training (gradients and scaler fit) |
| B | 60–80% | 8,765 | 8,595 | Checkpoint selection, calibration |
| C | 80–100% | 8,765 | 8,595 | Final test only |

The configuration lists both 0.6 and 0.8, but the pinned `src/train.py` only
splits at `train_split = 0.6`. As a result, B and C together decide early
stopping, and C is later used for testing. This applies to the original-code
checkpoint. The strict runner keeps everything else unchanged and lets only B
select the checkpoint.

## Setup

Clone recursively, so that both the paper source and its nested data submodule
are fetched:

```powershell
git clone --recurse-submodules https://github.com/minquanle/ProLoaF_.git
cd ProLoaF_
conda create -n proloaf-paper-repro python=3.9 -y
conda activate proloaf-paper-repro
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-deps -e .\paper_source
```

To use the original notebooks, also register the kernel:

```powershell
python -m ipykernel install --user --name proloaf-paper-repro --display-name "Python (ProLoaF Paper Reproduction)"
```

All commands below run from the repository root, in the
`proloaf-paper-repro` environment.

## Reproducing the results

The finished outputs of every step are already committed under
`artifacts/paper_reproduction/`. Re-running is only needed to check them.
Training runners refuse to write into an existing output directory, so give
them a new one.

**1. Original-code LSTM.** This uses the upstream code unchanged, with
`torch.manual_seed(1)`, Gaussian NLL loss and early-stopping patience 7.
Early stopping happened at epoch 41, and the weights from epoch 34 were kept.

```powershell
cd paper_source
python .\src\train.py -s opsd
cd ..
```

**2. Strict LSTM, three seeds.** Seed 1 took about 27 minutes on four CPU
threads.

```powershell
python .\reproduction\run_lstm_strict_602020.py --output-dir .\artifacts\paper_reproduction\strict_new_seed1 --seed 1 --threads 4
python .\reproduction\run_lstm_strict_602020.py --output-dir .\artifacts\paper_reproduction\strict_new_seed2 --seed 2 --threads 4
python .\reproduction\run_lstm_strict_602020.py --output-dir .\artifacts\paper_reproduction\strict_new_seed3 --seed 3 --threads 4
python .\reproduction\aggregate_lstm_seeds.py
```

Add `--dry-run` to check the split and tensor shapes without training.
`aggregate_lstm_seeds.py` reads the committed folders `strict_602020_20261013`,
`strict_602020_seed2` and `strict_602020_seed3`.

**3. Baselines and Prophet.**

```powershell
python .\reproduction\run_two_baselines.py
python .\reproduction\run_prophet_paper_protocol.py
```

The Prophet runner writes two variants:

- `paper_literal_80pct` keeps the notebook's default 80% interval and its
  147-hour timestamp offset. It is evidence of source fidelity only.
- `chronological_aligned_95pct` aligns forecasts with actuals and uses a 95%
  interval. This is the variant used in the table above.

**4. Interval calibration.** This runs on the original-code checkpoint.

```powershell
python .\reproduction\calibrate_lstm_intervals.py
python .\reproduction\evaluate_calibrated_lstm_on_test.py
```

The calibration works in three steps:

1. On B, compute r = |y − μ| / (1.96σ) for every forecast–actual pair.
2. Take c as the empirical 95th percentile of r.
3. Freeze c and apply it unchanged to C.

**5. Residual histogram (Fig. 3 of the paper).**

```powershell
python .\reproduction\plot_residual_comparison.py
```

**6. DLinear.**

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduction\setup_dlinear_source.ps1
python .\reproduction\run_dlinear_opsd.py --output-dir .\artifacts\paper_reproduction\dlinear_new_run
python .\reproduction\verify_dlinear_opsd.py --run-dir .\artifacts\paper_reproduction\dlinear_new_run
```

The full protocol is in
[`reproduction/DLINEAR_PROTOCOL.md`](reproduction/DLINEAR_PROTOCOL.md).

**7. Demo.**

```powershell
python .\demo\run_demo.py
```

Then open <http://127.0.0.1:8501>. The dashboard replays the 8,595 test origins
of 2019. It shows the 147-hour history, the next 24 actual hours, the LSTM
forecast, and both the original and the calibrated 95% intervals.

The first page load prepares all forecasts, which takes about 10 seconds. The
demo only runs inference. It uses the original-code checkpoint and c =
0.841887, not the strict LSTM.

## Metrics

- **RMSE and MAE** measure point-forecast error.
- **PICP** is the share of actual values inside the interval.
- **Width** is the mean interval width (called Sharpness in ProLoaF).
- **MIS** is the standard mean interval score for α = 0.05: the interval width
  plus 40 × the distance by which an actual value falls outside the interval.

The upstream `Mis.from_quantiles` includes the median when selecting α. It
therefore does not score the displayed 95% interval. Its value is kept
separately as `ProLoaF_legacy_MIS` and is not used in the table above.

## Limitations

- Three seeds are few, given how much the LSTM varies across them. The
  LSTM–DLinear gap is not statistically established.
- The interval calibration, the residual histogram and the demo still use the
  original-code checkpoint.
- Austrian load is not scaled by the upstream configuration, so it enters the
  network in raw MW (about 7,000). This is kept as in the source. It may relate
  to the high horizon-1 error, but this has not been tested.
- The paper reports no RMSE, MAE or PICP table. Only Fig. 3 (residual
  histogram) can be compared, and the redrawn figure does not match it
  exactly.
- GRU and other ProLoaF model types were not trained.

## Not included in this repository

- The submission report and presentation slides are kept as local
  deliverables.
- The raw OPSD download and earlier exploratory notebooks are not included.
  The experiments use only `paper_source/data/opsd.csv`.
- The DLinear source checkout (`external/LTSF-Linear/`) is not included.
  `setup_dlinear_source.ps1` restores it at commit `0c11366`.
- Runtime-only files are not included: TensorBoard runs of the strict
  experiments, per-run `best_weights.pt` copies and console logs. Every
  committed `opsd_recurrent.pkl` contains the selected weights.

## Provenance

[`reproduction/REPRODUCTION_PROTOCOL.md`](reproduction/REPRODUCTION_PROTOCOL.md)
and [`reproduction/RUN_CARD.md`](reproduction/RUN_CARD.md) record source
commits, environment and file hashes.
[`artifacts/paper_reproduction/README.md`](artifacts/paper_reproduction/README.md)
describes each result folder.

## References

- G. Gürses-Tran, F. Oppermann, A. Monti. *ProLoaF: Probabilistic load
  forecasting for power systems.* SoftwareX 23 (2023) 101487.
  <https://doi.org/10.1016/j.softx.2023.101487>
- Source code: <https://github.com/ElsevierSoftwareX/SOFTX-D-21-00190>
  (commit `1055d1fd3ae00ffce0b67eed03ec84a7410f5da8`).
- A. Zeng, M. Chen, L. Zhang, Q. Xu. *Are Transformers Effective for Time
  Series Forecasting?* AAAI 2023. <https://doi.org/10.1609/aaai.v37i9.26317>;
  code <https://github.com/cure-lab/LTSF-Linear> (commit `0c11366`).
- Open Power System Data, Time Series 2020-10-06 and Weather Data 2020-09-16.
  <https://data.open-power-system-data.org/>
