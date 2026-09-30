"""Evaluate the validation-frozen LSTM interval multiplier once on the OPSD test set.

This script is intentionally separate from ``calibrate_lstm_intervals.py``:
it reads the already selected c from JSON and never searches or optimizes c.
It therefore provides an honest final test evaluation for the post-hoc
uncertainty-calibration extension.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from calibrate_lstm_intervals import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_ROOT,
    calculate_metrics,
    metrics_by_horizon,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    calibration_dir = args.calibration_dir.resolve()
    calibration_path = calibration_dir / "selected_calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError("Run calibrate_lstm_intervals.py before test evaluation: {}".format(calibration_path))

    with calibration_path.open(encoding="utf-8") as file_handle:
        calibration = json.load(file_handle)
    if calibration.get("test_set_used") is not False:
        raise ValueError("Calibration metadata must state that the test set was not used to select c.")
    selected_c = float(calibration["selected_c"])

    source_code = source_root / "src"
    checkpoint = source_root / "oracles" / "opsd_recurrent.pkl"
    config_path = source_root / "targets" / "opsd" / "config.json"
    data_path = source_root / "data" / "opsd.csv"
    for required_path in (source_code, checkpoint, config_path, data_path):
        if not required_path.exists():
            raise FileNotFoundError("Required paper-reproduction input not found: {}".format(required_path))

    sys.path.insert(0, str(source_code))
    import proloaf.datahandler as dh  # pylint: disable=import-outside-toplevel
    import proloaf.tensorloader as tl  # pylint: disable=import-outside-toplevel
    import proloaf.modelhandler as mh  # pylint: disable=import-outside-toplevel
    import proloaf.metrics as paper_metrics  # pylint: disable=import-outside-toplevel

    with config_path.open(encoding="utf-8") as file_handle:
        config = json.load(file_handle)
    frame = pd.read_csv(data_path, sep=";", index_col=0, parse_dates=True)
    if config.get("target_list") is not None:
        frame[config["target_id"]] = frame[config["target_list"]].sum(axis=1)

    # This is exactly the hold-out partition used by src/evaluate.py: [80%, 100%).
    _pre_test_frame, test_frame = dh.split(frame, [config["validation_split"]])
    del _pre_test_frame

    with torch.no_grad():
        model = mh.ModelHandler.load_model(str(checkpoint), locate="cpu")
        model.to("cpu")
        model.model.eval()
        test_data = tl.TimeSeriesData(
            test_frame,
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
        loader = test_data.make_data_loader(batch_size=None, shuffle=False)
        inputs_enc, inputs_dec, targets = next(iter(loader))
        raw_prediction = model.predict(inputs_enc, inputs_dec)
        original_quantiles = model.loss_metric.get_quantile_prediction(
            predictions=raw_prediction, target=targets
        )
        # Cross-check the manually computed interval metrics against the actual
        # ProLoaF metric implementation before writing reported results.
        source_picp = float(
            paper_metrics.Picp().from_quantiles(targets, original_quantiles, avg_over="all").item()
        )
        source_sharpness = float(
            paper_metrics.Sharpness().from_quantiles(targets, original_quantiles, avg_over="all").item()
        )
        source_mis = float(
            paper_metrics.Mis().from_quantiles(targets, original_quantiles, avg_over="all").item()
        )
        source_bounds = original_quantiles.select_quantiles((0.975, 0.025)).values
        source_mis_direct = float(
            paper_metrics.Mis.func(targets, source_bounds, alpha=0.05, avg_over="all").item()
        )

    actual = targets[:, :, 0].detach().cpu().numpy()
    mean = raw_prediction[:, :, 0].detach().cpu().numpy()
    sigma = torch.exp(0.5 * raw_prediction[:, :, 1]).detach().cpu().numpy()
    if not (np.isfinite(actual).all() and np.isfinite(mean).all() and np.isfinite(sigma).all()):
        raise ValueError("Test prediction contains non-finite values.")

    summary = pd.DataFrame(
        [
            {"interval": "original_paper_checkpoint", **calculate_metrics(actual, mean, sigma, 1.0)},
            {"interval": "validation_calibrated_frozen_c", **calculate_metrics(actual, mean, sigma, selected_c)},
        ]
    )
    # The paper's Mis.from_quantiles includes the median (0.5) when it chooses
    # alpha, which causes it to score a degenerate 100% interval instead of its
    # displayed 95% interval.  We report the standard 95% score above and keep
    # the legacy value separately only for an auditable comparison to evaluate.py.
    summary["ProLoaF_legacy_MIS"] = 2.0 * summary["MAE"]
    original_row = summary.loc[summary["interval"] == "original_paper_checkpoint"].iloc[0]
    if not (
        np.isclose(original_row["PICP_percent"], source_picp, rtol=1e-5)
        and np.isclose(original_row["Sharpness"], source_sharpness, rtol=1e-5)
        and np.isclose(original_row["MIS_standard_95"], source_mis_direct, rtol=1e-5)
        and np.isclose(original_row["ProLoaF_legacy_MIS"], source_mis, rtol=1e-5)
    ):
        raise AssertionError(
            "Metric cross-check failed: PICP={:.8f}, Sharpness={:.8f}, standard_MIS={:.8f}; "
            "source PICP={:.8f}, Sharpness={:.8f}, legacy_MIS={:.8f}, direct_standard_MIS={:.8f}".format(
                original_row["PICP_percent"], original_row["Sharpness"], original_row["MIS_standard_95"],
                source_picp, source_sharpness, source_mis, source_mis_direct,
            )
        )
    horizon = pd.concat(
        [
            metrics_by_horizon(actual, mean, sigma, 1.0, "original_paper_checkpoint"),
            metrics_by_horizon(actual, mean, sigma, selected_c, "validation_calibrated_frozen_c"),
        ],
        ignore_index=True,
    )
    summary.to_csv(calibration_dir / "test_calibrated_summary.csv", index=False)
    horizon.to_csv(calibration_dir / "test_calibrated_metrics_by_horizon.csv", index=False)

    print("Frozen validation multiplier evaluated once on the held-out test set.")
    print("Selected c from validation = {:.8f}".format(selected_c))
    print("ProLoaF metric cross-check passed (PICP, Sharpness, MIS).")
    print(summary.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
    print("Artifacts written to {}".format(calibration_dir))


if __name__ == "__main__":
    main()
