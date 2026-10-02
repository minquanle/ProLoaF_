r"""Calibrate the existing ProLoaF LSTM 95% prediction interval on validation.

This is a *post-hoc* experiment for the midterm, not a modification of the
paper's model or training code.  It deliberately:

* loads the already trained ``opsd_recurrent.pkl`` checkpoint;
* keeps the LSTM mean forecast (mu) unchanged;
* uses only the original validation period (60%--80% of OPSD) to choose c;
* never loads the test partition.

The original Gaussian interval is ``mu +/- z_0.975 * sigma``.  We assess
``mu +/- z_0.975 * c * sigma`` and select c so validation empirical coverage
is 95%.  Therefore c < 1 narrows an over-conservative interval.

Run from any PowerShell directory with the paper reproduction environment:
    python .\reproduction\calibrate_lstm_intervals.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from statistics import NormalDist
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "paper_source"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "artifacts" / "paper_reproduction" / "opsd_recurrent.pkl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "paper_reproduction"
    / "interval_calibration"
)
TARGET_COVERAGE = 95.0
ALPHA = 0.05
Z_975 = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root of the frozen paper-reproduction source and checkpoint.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Folder receiving calibration-only artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained OPSD LSTM checkpoint.",
    )
    return parser.parse_args()


def calculate_metrics(actual: np.ndarray, mean: np.ndarray, sigma: np.ndarray, c: float) -> Dict[str, float]:
    """Return the same interval quantities used by ProLoaF for one c value."""
    lower = mean - Z_975 * c * sigma
    upper = mean + Z_975 * c * sigma
    in_interval = (actual > lower) & (actual <= upper)  # identical boundary convention to metrics.Picp
    width = upper - lower
    interval_score = width + (2.0 / ALPHA) * np.maximum(lower - actual, 0.0) + (2.0 / ALPHA) * np.maximum(actual - upper, 0.0)
    error = actual - mean
    return {
        "c": float(c),
        "PICP_percent": float(100.0 * in_interval.mean()),
        "abs_coverage_error_pp": float(abs(100.0 * in_interval.mean() - TARGET_COVERAGE)),
        "Sharpness": float(width.mean()),
        # Standard 95% mean interval score: width plus an alpha=0.05 penalty
        # for observations outside the interval.
        "MIS_standard_95": float(interval_score.mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "MAE": float(np.mean(np.abs(error))),
    }


def metrics_by_horizon(actual: np.ndarray, mean: np.ndarray, sigma: np.ndarray, c: float, label: str) -> pd.DataFrame:
    """Make the 24-step table needed to inspect calibration across the forecast horizon."""
    lower = mean - Z_975 * c * sigma
    upper = mean + Z_975 * c * sigma
    score = (upper - lower) + (2.0 / ALPHA) * np.maximum(lower - actual, 0.0) + (2.0 / ALPHA) * np.maximum(actual - upper, 0.0)
    return pd.DataFrame(
        {
            "interval": label,
            "c": c,
            "horizon_hour": np.arange(1, actual.shape[1] + 1),
            "PICP_percent": 100.0 * ((actual > lower) & (actual <= upper)).mean(axis=0),
            "Sharpness": (upper - lower).mean(axis=0),
            "MIS_standard_95": score.mean(axis=0),
            "RMSE": np.sqrt(np.mean(np.square(actual - mean), axis=0)),
            "MAE": np.mean(np.abs(actual - mean), axis=0),
        }
    )


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_code = source_root / "src"
    config_path = source_root / "targets" / "opsd" / "config.json"
    data_path = source_root / "data" / "opsd.csv"
    for required_path in (source_code, checkpoint, config_path, data_path):
        if not required_path.exists():
            raise FileNotFoundError("Required paper-reproduction input not found: {}".format(required_path))

    # Import the exact ProLoaF implementation only after the source path is known.
    sys.path.insert(0, str(source_code))
    import proloaf.datahandler as dh  # pylint: disable=import-outside-toplevel
    import proloaf.tensorloader as tl  # pylint: disable=import-outside-toplevel
    import proloaf.modelhandler as mh  # pylint: disable=import-outside-toplevel

    with config_path.open(encoding="utf-8") as file_handle:
        config = json.load(file_handle)
    frame = pd.read_csv(data_path, sep=";", index_col=0, parse_dates=True)
    if config.get("target_list") is not None:
        frame[config["target_id"]] = frame[config["target_list"]].sum(axis=1)

    # This matches train.py exactly: train is [0, 0.6), validation is [0.6, 1.0).
    # We then retain [0.6, 0.8), the portion called validation in the project plan.
    train_frame, remaining_frame = dh.split(frame, [config["train_split"]])
    validation_fraction_of_remaining = (
        (config["validation_split"] - config["train_split"])
        / (1.0 - config["train_split"])
    )
    validation_frame, _unused_test_frame = dh.split(remaining_frame, [validation_fraction_of_remaining])
    del train_frame, remaining_frame, _unused_test_frame  # Explicitly ensure no test data enters prediction.

    with torch.no_grad():
        model = mh.ModelHandler.load_model(str(checkpoint), locate="cpu")
        model.to("cpu")
        model.model.eval()
        validation_data = tl.TimeSeriesData(
            validation_frame,
            device="cpu",
            preparation_steps=[
                partial(dh.set_to_hours, freq="1H"),
                partial(dh.fill_if_missing, periodicity=config.get("periodicity", 24)),
                dh.add_cyclical_features,
                dh.add_onehot_features,
                model.scalers.transform,
                dh.check_continuity,
            ],
            **config,
        )
        loader = validation_data.make_data_loader(batch_size=None, shuffle=False)
        inputs_enc, inputs_dec, targets = next(iter(loader))
        raw_prediction = model.predict(inputs_enc, inputs_dec)

    actual = targets[:, :, 0].detach().cpu().numpy()
    mean = raw_prediction[:, :, 0].detach().cpu().numpy()
    sigma = torch.exp(0.5 * raw_prediction[:, :, 1]).detach().cpu().numpy()
    if not (np.isfinite(actual).all() and np.isfinite(mean).all() and np.isfinite(sigma).all()):
        raise ValueError("Validation prediction contains non-finite values.")
    if np.any(sigma <= 0):
        raise ValueError("Gaussian standard deviations must be positive.")

    # Exact empirical calibration: the 95th percentile is the smallest multiplier
    # that covers 95% of validation residuals under a symmetric Gaussian interval.
    normalized_residual = np.abs(actual - mean) / (Z_975 * sigma)
    selected_c = float(np.quantile(normalized_residual, TARGET_COVERAGE / 100.0))

    # A visible sweep accompanies the closed-form result, so the choice can be audited.
    grid = np.unique(
        np.concatenate((np.arange(0.50, 1.201, 0.005), np.array([1.0, selected_c])))
    )
    grid_metrics = pd.DataFrame(
        [calculate_metrics(actual, mean, sigma, float(c)) for c in grid]
    ).sort_values("c").reset_index(drop=True)
    selection_order = np.lexsort(
        (grid_metrics["MIS_standard_95"].to_numpy(), grid_metrics["abs_coverage_error_pp"].to_numpy())
    )
    selection_rank = np.empty(len(grid_metrics), dtype=int)
    selection_rank[selection_order] = np.arange(1, len(grid_metrics) + 1)
    grid_metrics["selection_rank"] = selection_rank

    uncalibrated = calculate_metrics(actual, mean, sigma, c=1.0)
    calibrated = calculate_metrics(actual, mean, sigma, c=selected_c)
    summary = pd.DataFrame(
        [
            {"interval": "original_paper_checkpoint", **uncalibrated},
            {"interval": "validation_calibrated", **calibrated},
        ]
    )
    horizon = pd.concat(
        [
            metrics_by_horizon(actual, mean, sigma, 1.0, "original_paper_checkpoint"),
            metrics_by_horizon(actual, mean, sigma, selected_c, "validation_calibrated"),
        ],
        ignore_index=True,
    )

    grid_metrics.to_csv(output_dir / "calibration_grid.csv", index=False)
    summary.to_csv(output_dir / "validation_calibration_summary.csv", index=False)
    horizon.to_csv(output_dir / "validation_metrics_by_horizon.csv", index=False)
    metadata = {
        "purpose": "Post-hoc standard-deviation scaling selected on validation only",
        "test_set_used": False,
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "data": str(data_path),
        "validation_period": "original OPSD rows [60%, 80%)",
        "validation_windows": int(actual.shape[0]),
        "forecast_horizon_hours": int(actual.shape[1]),
        "confidence_level_percent": 95.0,
        "z_0_975": Z_975,
        "formula": "lower = mu - z_0.975 * c * sigma; upper = mu + z_0.975 * c * sigma",
        "selection_rule": "c = empirical 95th percentile of abs(y-mu)/(z_0.975*sigma) on validation",
        "mis_definition": "standard 95% mean interval score, not ProLoaF's legacy Mis.from_quantiles output",
        "selected_c": selected_c,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "selected_calibration.json").open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(grid_metrics["c"], grid_metrics["PICP_percent"], color="#1f77b4")
    axes[0].axhline(TARGET_COVERAGE, color="#d62728", linestyle="--", label="target 95%")
    axes[0].axvline(selected_c, color="#2ca02c", linestyle="--", label="selected c")
    axes[0].set(xlabel="standard-deviation multiplier c", ylabel="PICP (%)", title="Coverage on validation")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(grid_metrics["c"], grid_metrics["Sharpness"], color="#9467bd", label="Sharpness")
    axes[1].plot(grid_metrics["c"], grid_metrics["MIS_standard_95"], color="#ff7f0e", label="standard 95% MIS")
    axes[1].axvline(selected_c, color="#2ca02c", linestyle="--", label="selected c")
    axes[1].set(xlabel="standard-deviation multiplier c", title="Width--score trade-off")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.suptitle("Validation-only calibration of ProLoaF LSTM intervals")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_calibration_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    print("Validation calibration complete. Test set was not used.")
    print(summary.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
    print("Selected c = {:.8f}".format(selected_c))
    print("Artifacts written to {}".format(output_dir))


if __name__ == "__main__":
    main()
