# LSTM experiment with a strict chronological 60/20/20 split

This is a new experiment, separate from the preserved upstream-code reproduction.
It does not replace the old checkpoint, demo, report, slides, or baseline results.

## Protocol

- Same author-prepared dataset: 43,824 hourly rows, 2015–2019.
- A: first 26,294 hours; gradients and scaler fitting only.
- B: next 8,765 hours; validation loss and checkpoint selection only.
- C: final 8,765 hours; final evaluation after checkpoint selection.
- Same frozen configuration, LSTM architecture, seed 1, Adam learning rate,
  batch size 32, maximum 50 epochs, early-stopping patience 7.
- Existing upstream training loop and data loaders are reused. This includes
  its validation batching/shuffling/drop-last behavior.
- Four CPU threads are explicitly configured. The earlier run did not record
  thread count, so bitwise-equivalent training is not claimed.
- The best B-loss weights are restored even if the run reaches 50 epochs.
  Upstream restores best weights automatically only on early stopping.
- No old model is loaded for training or compared during checkpoint selection.
- Original Gaussian 95% intervals are evaluated with c=1. The previous
  calibration multiplier 0.841887 is not applied to this new model.
- B also guides model selection, so it is not an independent calibration set.
  No additional calibration is performed in this experiment.

## Artifacts and status

`status.json` is the authoritative progress record. Results are available only
when its state is `complete`; `failed` contains the error trace.

- `protocol.json`: boundaries, hashes, runtime, and exclusion of C from selection.
- `epoch_history.csv`: full-precision A training loss and B validation loss.
- `best_weights.pt`: durable copy of the B-selected parameter state.
- `opsd_recurrent.pkl`: new checkpoint including scaler and configuration.
- `test_predictions.npz`: all 8,595 × 24 final predictions and Gaussian sigmas.
- `paper_metrics.csv`: legacy upstream metric definitions for comparison.
- `test_summary.csv`: original intervals, point metrics, and standard 95% MIS.
- `test_metrics_by_horizon.csv`: metrics for horizons 1–24.
- `comparison_old_vs_strict.csv`: old and new results on the same C windows.

The old result was selected using B+C, so differences do not constitute a
controlled comparison of two equally independent test protocols. A lower score
is not guaranteed by fixing the split. Report the observed result, including
degradation if any. The upstream legacy MIS is kept separate from standard MIS.

## Re-run

Use a fresh output directory, since the runner refuses overwriting an existing run:

```powershell
conda activate proloaf-paper-repro
python .\reproduction\run_lstm_strict_602020.py --output-dir .\artifacts\paper_reproduction\strict_602020_new_run --threads 4
```

Console logs for the current background run are located one directory above,
as `strict_602020_20261013.stdout.log` and `strict_602020_20261013.stderr.log`.
