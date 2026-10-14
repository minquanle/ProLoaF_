"""Rebuild point forecasts and plot residuals on exactly the aligned test pairs.

No training or interval calibration is performed. Cached predictions include a
checkpoint hash so rerunning this script cannot silently use another model.
"""
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/paper_reproduction/residual_comparison"
CHECKPOINT = ROOT / "artifacts/paper_reproduction/opsd_recurrent.pkl"
PROPHET = ROOT / "artifacts/paper_reproduction/prophet_paper_protocol/prophet_continuous_test_forecast_95_MW.csv"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    fingerprint = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    sys.path.insert(0, str(ROOT / "paper_source/src"))
    import proloaf.datahandler as dh
    import proloaf.tensorloader as tl
    import proloaf.modelhandler as mh
    config = json.loads((ROOT / "artifacts/paper_reproduction/training_config_used.json").read_text(encoding="utf-8"))
    # The persisted scaler, not test extrema, defines conversion back to MW.
    model = mh.ModelHandler.load_model(str(CHECKPOINT), locate="cpu")
    model.to("cpu")
    model.model.eval()
    frame = pd.read_csv(ROOT / "paper_source/data/opsd.csv", sep=";", index_col=0, parse_dates=True)
    if config.get("target_list") is not None:
        frame[config["target_id"]] = frame[config["target_list"]].sum(axis=1)
    _, test = dh.split(frame, [config["validation_split"]])
    cache = OUT / "lstm_point_predictions.npz"
    if cache.exists():
        saved = np.load(cache)
        if saved["checkpoint_sha256"].item() != fingerprint:
            raise ValueError("Checkpoint changed; remove the explicit cache before rerunning.")
        actual, mean = saved["actual_scaled"], saved["mean_scaled"]
    else:
        data = tl.TimeSeriesData(test, device="cpu", preparation_steps=[
            partial(dh.set_to_hours, freq="1H"),
            partial(dh.fill_if_missing, periodicity=config.get("periodicity", 24)),
            dh.add_cyclical_features, dh.add_onehot_features,
            model.scalers.transform, dh.check_continuity], **config)
        values, predictions = [], []
        loader = data.make_data_loader(batch_size=256, shuffle=False, drop_last=False)
        with torch.no_grad():
            for i, (enc, dec, targets) in enumerate(loader):
                pred = model.predict(enc, dec)
                values.append(targets[:, :, 0].cpu().numpy())
                predictions.append(pred[:, :, 0].cpu().numpy())
                if i % 8 == 0:
                    print("Inference: {} windows".format(sum(len(x) for x in values)), flush=True)
        actual, mean = np.concatenate(values), np.concatenate(predictions)
        np.savez_compressed(cache, actual_scaled=actual, mean_scaled=mean, checkpoint_sha256=fingerprint)
    # Original persisted MinMax fit: min=31307 MW and range=46242 MW.
    scale = 46242.0
    minimum = 31307.0
    n, h = actual.shape
    assert (n, h) == (8595, 24), actual.shape
    indices = np.arange(n)[:, None] + config["history_horizon"] + np.arange(h)[None, :]
    timestamps = test.index.to_numpy()[indices]
    prophet = pd.read_csv(PROPHET, parse_dates=["timestamp"]).set_index("timestamp")
    forecast = prophet.loc[pd.DatetimeIndex(timestamps.ravel()), "prediction_MW"].to_numpy().reshape(n, h)
    target_id = config["target_id"][0] if isinstance(config["target_id"], list) else config["target_id"]
    truth_mw = test[target_id].to_numpy()[indices]
    np.testing.assert_allclose(actual * scale + minimum, truth_mw, atol=0.02, rtol=0)
    residuals = {"ProLoaF LSTM": (actual - mean).astype(np.float64).ravel(),
                 "Prophet aligned": ((truth_mw - forecast) / scale).ravel()}
    stats = {}
    for name, r in residuals.items():
        assert np.isfinite(r).all()
        stats[name] = {"n": int(r.size), "mean_MW": float(r.mean()*scale),
                       "std_MW": float(r.std()*scale), "MAE_MW": float(np.abs(r).mean()*scale),
                       "RMSE_MW": float(np.sqrt(np.mean(r*r))*scale)}
    np.testing.assert_allclose(stats["ProLoaF LSTM"]["RMSE_MW"], 3262.64077, atol=0.2)
    np.testing.assert_allclose(stats["Prophet aligned"]["RMSE_MW"], 3946.46765, atol=0.2)
    # Include every tail. Common bin edges are used for both local series.
    extrema = np.concatenate(list(residuals.values()))
    edges = np.linspace(np.floor(extrema.min()*100)/100, np.ceil(extrema.max()*100)/100, 61)
    counts = {name: np.histogram(r, edges)[0] for name, r in residuals.items()}
    assert all(v.sum() == n*h for v in counts.values())
    bins = pd.DataFrame({"left": edges[:-1], "right": edges[1:],
                         "lstm_count": counts["ProLoaF LSTM"], "prophet_count": counts["Prophet aligned"]})
    bins.to_csv(OUT / "histogram_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.7), constrained_layout=True)
    for (name, values), color in zip(counts.items(), ["#3183A7", "#E48E7E"]):
        ax.bar(edges[:-1], values, width=np.diff(edges), align="edge", color=color, alpha=0.67,
               edgecolor="white", linewidth=0.25, label=name)
    ax.axvline(0, color="#344054", linestyle="--", linewidth=1)
    ax.set(title="Distribution of residuals — local experiment",
           xlabel="Residual = (actual − predicted) / 46,242 MW", ylabel="Frequency (forecast pairs)")
    ax.set_axisbelow(True)
    ax.grid(alpha=0.2)
    ax.legend(framealpha=0.95)
    fig.savefig(OUT / "residual_histogram_local.png", dpi=250)
    plt.close(fig)
    # Author's notebook: Residuals(avg_over='time'), followed by plot_hist(bins=80).
    # Each origin contributes ONE signed mean error across its 24-hour horizon.
    origin_residuals = {name: r.reshape(n, h).mean(axis=1) for name, r in residuals.items()}
    import proloaf.metrics as paper_metrics
    reference = paper_metrics.Residuals.func(
        torch.tensor(actual[:, :, None]), torch.tensor(mean[:, :, None]), avg_over="time").numpy()
    np.testing.assert_allclose(origin_residuals["ProLoaF LSTM"], reference, atol=1e-7)
    origin_frame = pd.DataFrame(origin_residuals)
    origin_frame.to_csv(OUT / "residuals_mean24h_per_origin.csv", index=False)
    # Native author plotting function selects 80 automatic bins per column.
    import proloaf.plot as author_plot
    author_plot.plot_hist(origin_frame, bins=80)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.235)
    ax = fig.axes[0]
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("Frequency", fontsize=14)
    ax.tick_params(labelsize=12)
    # Match the published visible horizontal window and regular tick spacing.
    # This is a zoom, not a new residual definition or rescaling of counts.
    view_left, view_right = -0.235, 0.115
    ax.set_xlim(view_left, view_right)
    ax.set_xticks([-0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10])
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.set_ylim(0, 550)
    ax.set_facecolor("#E5E5E5")
    ax.grid(color="white", linewidth=0.6)
    omitted = {name: int(((r < view_left) | (r > view_right)).sum())
               for name, r in origin_residuals.items()}
    # Keep the paper's blue/orange distinction; do not alter bin heights.
    for i, patch in enumerate(ax.patches):
        patch.set_facecolor("#3183A7" if i < 80 else "#E48E7E")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#E48E7E", alpha=0.5, label="prophet"),
                       Patch(facecolor="#3183A7", alpha=0.5, label="proloaf_seq2seq")], fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "residual_histogram_author_method.png", dpi=250)
    # A separate full-range audit figure keeps every outlying residual visible.
    ax.set_xlim(combined_min := min(r.min() for r in origin_residuals.values()) - 0.01,
                max(r.max() for r in origin_residuals.values()) + 0.01)
    ax.set_xticks(np.arange(np.ceil(combined_min/0.05)*0.05, 0.1501, 0.05))
    fig.savefig(OUT / "residual_histogram_author_full_range.png", dpi=250)
    plt.close(fig)
    # For an editable side-by-side slide, use a common 80-bin grid explicitly.
    # This changes bin edges, not the per-origin residuals or their population.
    combined = np.concatenate(list(origin_residuals.values()))
    common_edges = np.linspace(combined.min(), combined.max(), 81)
    common_counts = {k: np.histogram(v, common_edges)[0] for k,v in origin_residuals.items()}
    assert all(v.sum() == n for v in common_counts.values())
    author_meta = {"n_per_model": n, "aggregation": "mean signed residual over 24 horizons per origin",
                  "published_style_view": {"x_limits": [view_left, view_right],
                                           "outside_view_per_model": omitted,
                                           "y_limits": [0, 550], "y_tick_spacing": 50},
                  "bins": 80, "author_png_bins": "80 automatic bins per series, exact source plot_hist",
                  "slide_bins": "80 common bins, for aligned editable columns",
                  "edges": common_edges.tolist(), "counts": {k:v.tolist() for k,v in common_counts.items()},
                  "source_notebook": "Train and Benchmark Models.ipynb, cells 61 and 66 (zero-based)",
                  "source_metric": "Residuals.func(avg_over='time')", "source_plot": "plot_hist(bins=80)",
                  "statistics": {k:{"mean_MW":float(v.mean()*scale), "std_MW":float(v.std()*scale),
                                    "min_scaled":float(v.min()), "max_scaled":float(v.max()),
                                    "peak_native_bin":int(common_counts[k].max())} for k,v in origin_residuals.items()},
                  "caveat": "This reproduces the aggregation in the pinned source, not proof of the exact published figure inputs."}
    (OUT / "author_method_summary.json").write_text(json.dumps(author_meta, indent=2), encoding="utf-8")
    comparison, panels = plt.subplots(1, 2, figsize=(16, 4.8))
    for panel, filename, title in zip(panels,
            ["paper_fig3_original.jpg", "residual_histogram_author_method.png"],
            ["Paper (Fig. 3)", "Local reproduction"]):
        panel.imshow(plt.imread(OUT / filename))
        panel.set_title(title, fontsize=15)
        panel.axis("off")
    comparison.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.12, wspace=0.02)
    comparison.text(0.5, 0.025,
        "Local view: 169 LSTM and 165 Prophet residuals outside x range. Counts are unchanged; local y scale extends to 550.",
        ha="center", fontsize=11)
    comparison.savefig(OUT / "paper_vs_local_matching_axes.png", dpi=200, facecolor="white")
    plt.close(comparison)
    print("Author aggregation (8,595 origins, not 206,280 individual errors):", author_meta["statistics"], flush=True)
    meta = {"residual_definition": "actual - predicted", "scale_MW": scale,
            "origins": n, "horizon": h, "forecast_pairs_per_model": n*h,
            "unique_target_timestamps": int(np.unique(timestamps).size),
            "target_start": str(timestamps.min()), "target_end": str(timestamps.max()),
            "checkpoint_sha256": fingerprint, "statistics": stats,
            "caveats": ["Overlapping horizons; pairs are not independent.",
                        "Prophet fit once on first 80%; LSTM uses rolling observed history and additional covariates.",
                        "C participated in early stopping in the pinned training source.",
                        "Paper Fig. 3 does not document sufficient aggregation details for exact histogram matching."],
            "bin_edges": edges.tolist(), "counts": {k: v.tolist() for k,v in counts.items()}}
    (OUT / "residual_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
