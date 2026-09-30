"""Evaluate two lightweight load-forecasting baselines fairly against ProLoaF.

This runner deliberately does not modify the frozen paper source.  It uses the
same OPSD file, chronological 60/20/20 split, 147-hour history, 24-hour horizon,
and Min-Max target scaling as the reproduced ProLoaF LSTM.

Baselines
---------
simple-naive
    Repeat the last observed load for all 24 forecast hours.
seasonal-naive
    Use the load from the same hour one day earlier.

Neither baseline is trained.  Their 95% prediction intervals are calibrated
from horizon-specific residual quantiles on the validation partition only;
the final test partition is never used to select those intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "paper_source" / "data" / "opsd.csv"
DEFAULT_CONFIG = ROOT / "paper_source" / "targets" / "opsd" / "config.json"
DEFAULT_LSTM_METRICS = (
    ROOT / "artifacts" / "paper_reproduction" / "evaluation" / "opsd_recurrent.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "paper_reproduction" / "baseline_comparison"

ALPHA = 0.05
SEASONALITY_HOURS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lstm-metrics", type=Path, default=DEFAULT_LSTM_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_chronologically(frame: pd.DataFrame, train_ratio: float, validation_ratio: float):
    train_end = int(len(frame) * train_ratio)
    validation_end = int(len(frame) * validation_ratio)
    return frame.iloc[:train_end], frame.iloc[train_end:validation_end], frame.iloc[validation_end:]


def make_windows(values: np.ndarray, history: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return actual targets plus the two point forecasts for every rolling origin."""
    required_rows = history + horizon
    if len(values) < required_rows:
        raise ValueError(f"Need at least {required_rows} rows, received {len(values)}.")

    n_origins = len(values) - history - horizon + 1
    actual = np.lib.stride_tricks.sliding_window_view(values[history:], horizon)
    actual = actual[:n_origins]

    # At an origin, values through index history - 1 are known.
    last_observed = values[history - 1 : history - 1 + n_origins]
    simple_naive = np.repeat(last_observed[:, None], horizon, axis=1)

    # For hour h in the forecast, use the same hour from the preceding day.
    seasonal_naive = np.lib.stride_tricks.sliding_window_view(
        values[history - SEASONALITY_HOURS :], horizon
    )[:n_origins]
    return actual, simple_naive, seasonal_naive


def interval_from_validation_errors(
    validation_actual: np.ndarray, validation_prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper residual offsets for a nominal 95% interval per horizon hour."""
    errors = validation_actual - validation_prediction
    return np.quantile(errors, ALPHA / 2, axis=0), np.quantile(errors, 1 - ALPHA / 2, axis=0)


def evaluate(
    actual: np.ndarray,
    prediction: np.ndarray,
    lower_offset: np.ndarray,
    upper_offset: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray]:
    lower = prediction + lower_offset[None, :]
    upper = prediction + upper_offset[None, :]
    errors = actual - prediction
    interval_width = upper - lower
    coverage = (actual >= lower) & (actual <= upper)

    below_lower = np.maximum(lower - actual, 0.0)
    above_upper = np.maximum(actual - upper, 0.0)
    interval_score = interval_width + (2 / ALPHA) * (below_lower + above_upper)

    overall = {
        "Mse": float(np.mean(errors**2)),
        "Rmse": float(np.sqrt(np.mean(errors**2))),
        "Mae": float(np.mean(np.abs(errors))),
        "Picp": float(np.mean(coverage) * 100),
        "Sharpness": float(np.mean(interval_width)),
        "Mis": float(np.mean(interval_score)),
        "Residuals": float(np.mean(errors)),
    }
    by_horizon = pd.DataFrame(
        {
            "horizon_hour": np.arange(1, actual.shape[1] + 1),
            "Mse": np.mean(errors**2, axis=0),
            "Rmse": np.sqrt(np.mean(errors**2, axis=0)),
            "Mae": np.mean(np.abs(errors), axis=0),
            "Picp": np.mean(coverage, axis=0) * 100,
            "Sharpness": np.mean(interval_width, axis=0),
            "Mis": np.mean(interval_score, axis=0),
            "Residuals": np.mean(errors, axis=0),
        }
    )
    return overall, by_horizon, lower, upper


def add_mw_columns(metrics: pd.DataFrame, target_range_mw: float) -> pd.DataFrame:
    result = metrics.copy()
    result["Mse_MW2"] = result["Mse"] * target_range_mw**2
    for metric in ("Rmse", "Mae", "Sharpness", "Mis", "Residuals"):
        result[f"{metric}_MW"] = result[metric] * target_range_mw
    return result


def load_lstm_row(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"LSTM evaluation CSV was not found: {path}. Run src/evaluate.py first."
        )
    frame = pd.read_csv(path, sep=";", index_col=0)
    row = frame.iloc[0]
    required = ("Mse", "Rmse", "Mae", "Picp", "Sharpness", "Mis", "Residuals")
    missing = [name for name in required if name not in row.index]
    if missing:
        raise ValueError(f"LSTM metrics CSV is missing columns: {missing}")
    return {name: float(row[name]) for name in required}


def main() -> None:
    args = parse_args()
    for path in (args.data, args.config, args.lstm_metrics):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    target = config["target_id"][0]
    history = int(config["history_horizon"])
    horizon = int(config["forecast_horizon"])
    train_ratio = float(config["train_split"])
    validation_ratio = float(config["validation_split"])

    frame = pd.read_csv(args.data, sep=";", index_col=0, parse_dates=True)
    if target not in frame.columns:
        raise ValueError(f"Target column not found: {target}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("The OPSD time index must be chronological.")
    if frame[target].isna().any():
        raise ValueError("The target contains missing values; stop rather than silently impute.")

    train, validation, test = split_chronologically(frame, train_ratio, validation_ratio)
    target_min = float(train[target].min())
    target_max = float(train[target].max())
    target_range = target_max - target_min
    if target_range <= 0:
        raise ValueError("Training target has zero range.")

    # Match the LSTM target's Min-Max scaling, fitted only on the training split.
    scale = lambda series: (series.to_numpy(dtype=float) - target_min) / target_range
    validation_values = scale(validation[target])
    test_values = scale(test[target])

    val_actual, val_simple, val_seasonal = make_windows(validation_values, history, horizon)
    test_actual, test_simple, test_seasonal = make_windows(test_values, history, horizon)
    if len(test_actual) != 8595:
        raise RuntimeError(f"Unexpected number of test origins: {len(test_actual)} (expected 8595).")

    models = {
        "simple-naive": (val_simple, test_simple, "Validation residual quantiles, 95%"),
        "seasonal-naive-24h": (val_seasonal, test_seasonal, "Validation residual quantiles, 95%"),
    }
    overall_rows: list[dict[str, object]] = []
    horizon_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    first_forecast = pd.DataFrame(
        {
            "timestamp": test.index[history : history + horizon],
            "actual_MW": test[target].iloc[history : history + horizon].to_numpy(),
        }
    )

    for name, (validation_prediction, test_prediction, interval_method) in models.items():
        started = time.perf_counter()
        lower_offset, upper_offset = interval_from_validation_errors(val_actual, validation_prediction)
        overall, by_horizon, lower, upper = evaluate(
            test_actual, test_prediction, lower_offset, upper_offset
        )
        elapsed = time.perf_counter() - started
        overall_rows.append(
            {
                "model": name,
                "interval_method": interval_method,
                "test_forecast_origins": len(test_actual),
                "forecast_horizon_hours": horizon,
                "runtime_seconds": elapsed,
                **overall,
            }
        )
        by_horizon.insert(0, "model", name)
        horizon_frames.append(by_horizon)
        calibration_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "horizon_hour": np.arange(1, horizon + 1),
                    "lower_error_quantile": lower_offset,
                    "upper_error_quantile": upper_offset,
                    "validation_forecast_origins": len(val_actual),
                }
            )
        )
        first_forecast[f"{name}_prediction_MW"] = test_prediction[0] * target_range + target_min
        first_forecast[f"{name}_lower_95_MW"] = lower[0] * target_range + target_min
        first_forecast[f"{name}_upper_95_MW"] = upper[0] * target_range + target_min

    lstm_metrics = load_lstm_row(args.lstm_metrics)
    overall_rows.insert(
        0,
        {
            "model": "ProLoaF-LSTM",
            "interval_method": "Gaussian NLL model output, 95%",
            "test_forecast_origins": len(test_actual),
            "forecast_horizon_hours": horizon,
            "runtime_seconds": np.nan,
            **lstm_metrics,
        },
    )

    overall = add_mw_columns(pd.DataFrame(overall_rows), target_range)
    per_horizon = pd.concat(horizon_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)

    overall.to_csv(args.output_dir / "two_baseline_comparison.csv", index=False)
    per_horizon.to_csv(args.output_dir / "two_baseline_metrics_by_horizon.csv", index=False)
    calibration.to_csv(args.output_dir / "two_baseline_interval_calibration.csv", index=False)
    first_forecast.to_csv(args.output_dir / "first_24h_baseline_forecast.csv", index=False)

    metadata = {
        "data_file": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "config_file": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "lstm_metrics_file": str(args.lstm_metrics.resolve()),
        "target": target,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "test_forecast_origins": len(test_actual),
        "history_horizon_hours": history,
        "forecast_horizon_hours": horizon,
        "target_train_min_mw": target_min,
        "target_train_max_mw": target_max,
        "target_train_range_mw": target_range,
        "interval_calibration": "Horizon-specific empirical 2.5% and 97.5% residual quantiles from validation only.",
        "note": "Baseline point forecasts use only values available at each rolling forecast origin.",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Two-baseline benchmark completed.")
    print(overall[["model", "Rmse", "Mae", "Picp", "Sharpness", "Mis"]].to_string(index=False))
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
