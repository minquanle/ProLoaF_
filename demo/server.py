"""Local HTTP backend for the polished ProLoaF forecast demonstration."""

from __future__ import annotations

import json
import sys
import threading
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import NormalDist
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO / "paper_source"
SOURCE_CODE = SOURCE_ROOT / "src"
CHECKPOINT = REPO / "artifacts" / "paper_reproduction" / "opsd_recurrent.pkl"
CONFIG_PATH = SOURCE_ROOT / "targets" / "opsd" / "config.json"
DATA_PATH = SOURCE_ROOT / "data" / "opsd.csv"
CALIBRATION_DIR = REPO / "artifacts" / "paper_reproduction" / "interval_calibration"
CALIBRATION_PATH = CALIBRATION_DIR / "selected_calibration.json"
TEST_SUMMARY_PATH = CALIBRATION_DIR / "test_calibrated_summary.csv"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TARGET = "DE_load_actual_entsoe_transparency"
Z_975 = NormalDist().inv_cdf(0.975)
# Final-exam model: 5-seed ensemble of the strict LSTM with scaled AT load and holiday flags.
FINAL_DIR = REPO / "artifacts" / "final_exam"
FINAL_ENSEMBLE = FINAL_DIR / "results" / "at_hol_ensemble_test.npz"
FINAL_TESTS = FINAL_DIR / "results" / "tests.json"
FINAL_SUMMARY = FINAL_DIR / "results" / "summary.csv"
FINAL_EXPLAIN_MODEL = FINAL_DIR / "runs" / "at_hol_seed2" / "model.pkl"
FINAL_CONFIG = FINAL_DIR / "runs" / "at_hol_seed1" / "training_config_used.json"
SHORT = {"AT_load_actual_entsoe_transparency": "Phụ tải Áo", TARGET: "Phụ tải Đức", "DE_temperature": "Nhiệt độ",
         "DE_radiation_direct_horizontal": "Bức xạ trực tiếp", "DE_radiation_diffuse_horizontal": "Bức xạ khuếch tán"}


class ForecastStore:
    def __init__(self) -> None:
        self.data = None
        self.error = None
        self.ready = threading.Event()

    def start_loading(self) -> None:
        threading.Thread(target=self._load, daemon=True, name="proloaf-demo-loader").start()

    def _load(self) -> None:
        try:
            for required in (SOURCE_CODE, CHECKPOINT, CONFIG_PATH, DATA_PATH, CALIBRATION_PATH):
                if not required.exists():
                    raise FileNotFoundError("Missing local artifact: {}".format(required))
            sys.path.insert(0, str(SOURCE_CODE))
            import proloaf.datahandler as dh  # pylint: disable=import-outside-toplevel
            import proloaf.tensorloader as tl  # pylint: disable=import-outside-toplevel
            import proloaf.modelhandler as mh  # pylint: disable=import-outside-toplevel

            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
            frame = pd.read_csv(DATA_PATH, sep=";", index_col=0, parse_dates=True)
            _before_test, test_frame = dh.split(frame, [config["validation_split"]])
            with torch.no_grad():
                model = mh.ModelHandler.load_model(str(CHECKPOINT), locate="cpu")
                model.to("cpu")
                model.model.eval()
                test_data = tl.TimeSeriesData(
                    test_frame, device="cpu",
                    preparation_steps=[partial(dh.set_to_hours, freq="1H"), partial(dh.fill_if_missing, periodicity=config.get("periodicity", 24)), dh.add_cyclical_features, dh.add_onehot_features, model.scalers.transform, dh.check_continuity],
                    **config,
                )
                inputs_enc, inputs_dec, targets = next(iter(test_data.make_data_loader(batch_size=None, shuffle=False)))
                raw_prediction = model.predict(inputs_enc, inputs_dec)
            target_scaler = model.scalers.scalers[TARGET]
            load_min, load_range = float(target_scaler.data_min_[0]), float(target_scaler.data_range_[0])
            midterm_summary = pd.read_csv(TEST_SUMMARY_PATH)
            self.models = {"midterm": {
                "label": "LSTM-gốc (giữa kỳ)",
                "actual": targets[:, :, 0].detach().cpu().numpy() * load_range + load_min,
                "mean": raw_prediction[:, :, 0].detach().cpu().numpy() * load_range + load_min,
                "sigma": torch.exp(0.5 * raw_prediction[:, :, 1]).detach().cpu().numpy() * load_range,
                "c": float(calibration["selected_c"]),
                "summary": [{"name": "LSTM-gốc, khoảng gốc", "c": 1.0, "picp": float(midterm_summary["PICP_percent"][0]),
                             "width": float(midterm_summary["Sharpness"][0]) * load_range,
                             "rmse": float(midterm_summary["RMSE"][0]) * load_range, "mae": float(midterm_summary["MAE"][0]) * load_range},
                            {"name": "LSTM-gốc, đã hiệu chỉnh", "c": float(midterm_summary["c"][1]), "picp": float(midterm_summary["PICP_percent"][1]),
                             "width": float(midterm_summary["Sharpness"][1]) * load_range,
                             "rmse": float(midterm_summary["RMSE"][1]) * load_range, "mae": float(midterm_summary["MAE"][1]) * load_range}],
            }}
            self.test_frame = test_frame
            self.history, self.horizon = int(config["history_horizon"]), int(config["forecast_horizon"])
            self._load_final(dh, tl, mh, frame, load_min, load_range)
            self.data = True
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.ready.set()

    def _load_final(self, dh, tl, mh, frame, load_min, load_range) -> None:
        """Final-exam ensemble from saved predictions, plus one seed for live Integrated Gradients."""
        sys.path.insert(0, str(REPO / "reproduction" / "final"))
        from run_lstm_variant import add_calendar_flags  # pylint: disable=import-outside-toplevel

        saved = np.load(FINAL_ENSEMBLE)
        tests = json.loads(FINAL_TESTS.read_text(encoding="utf-8"))
        summary = pd.read_csv(FINAL_SUMMARY)
        rows = summary[summary["variant"] == "at_hol"].set_index("model")
        c = float(tests["at_hol"]["c_global_from_B"])
        self.models["final"] = {
            "label": "LSTM-cuối kỳ (ensemble 5 seed)",
            "actual": saved["actual_scaled"] * load_range + load_min,
            "mean": saved["mean_scaled"] * load_range + load_min,
            "sigma": saved["sigma_scaled"] * load_range, "c": c,
            "summary": [{"name": name, "c": cc, "picp": float(rows.loc[key, "PICP_percent"]), "width": float(rows.loc[key, "Width_MW"]),
                         "rmse": float(rows.loc[key, "RMSE_MW"]), "mae": float(rows.loc[key, "MAE_MW"])}
                        for name, key, cc in [("LSTM-cuối kỳ, khoảng gốc", "ensemble raw", 1.0),
                                              ("LSTM-cuối kỳ, đã hiệu chỉnh", "ensemble + global c", c)]],
        }
        assert np.allclose(self.models["final"]["actual"], self.models["midterm"]["actual"], atol=0.05)
        config = json.loads(FINAL_CONFIG.read_text(encoding="utf-8"))
        flagged = add_calendar_flags(frame)
        _before, test_flagged = dh.split(flagged, [0.8])
        model = mh.ModelHandler.load_model(str(FINAL_EXPLAIN_MODEL), locate="cpu")
        model.to("cpu")
        self.explain_net = model.model_wrap.model if hasattr(model, "model_wrap") else model.model
        self.explain_net.eval()
        data = tl.TimeSeriesData(test_flagged, device="cpu", preparation_steps=[
            partial(dh.set_to_hours, freq="1H"), partial(dh.fill_if_missing, periodicity=24), dh.add_cyclical_features,
            dh.add_onehot_features, model.scalers.transform, dh.check_continuity], **config)
        self.explain_enc, self.explain_dec, _ = next(iter(data.make_data_loader(batch_size=None, shuffle=False)))
        self.enc_base = self.explain_enc.mean(0, keepdim=True)
        self.dec_base = self.explain_dec.mean(0, keepdim=True)
        self.enc_names = [SHORT[n] for n in config["encoder_features"]]
        self.dec_names = config["decoder_features"]
        self.holiday = ((test_flagged["is_holiday"] + test_flagged["is_xmas"]) > 0).to_numpy()
        self.load_range = load_range

    def explain(self, origin: int) -> dict:
        """Integrated Gradients (32 steps) of the summed 24 h mean for one forecast window."""
        enc, dec = self.explain_enc[origin:origin + 1], self.explain_dec[origin:origin + 1]
        total_e, total_d = torch.zeros_like(enc), torch.zeros_like(dec)
        for alpha in (torch.arange(32, dtype=torch.float32) + 0.5) / 32:
            e = (self.enc_base + alpha * (enc - self.enc_base)).requires_grad_(True)
            d = (self.dec_base + alpha * (dec - self.dec_base)).requires_grad_(True)
            ge, gd = torch.autograd.grad(self.explain_net(e, d)[0][:, :, 0].sum(), [e, d])
            total_e += ge
            total_d += gd
        attr_e = ((enc - self.enc_base) * total_e / 32)[0].detach().numpy() * self.load_range / 24
        attr_d = ((dec - self.dec_base) * total_d / 32)[0].detach().numpy() * self.load_range / 24
        groups = {"Lịch: giờ": "hour_", "Lịch: thứ": "weekday_", "Lịch: tháng": "month_"}
        parts = {SHORT_NAME: float(attr_e[:, i].sum()) for i, SHORT_NAME in enumerate(self.enc_names)}
        for label, prefix in groups.items():
            cols = [i for i, n in enumerate(self.dec_names) if n.startswith(prefix) and not n.endswith("sin")]
            parts[label] = float(attr_d[:, cols].sum())
        parts["Lịch: sin"] = float(attr_d[:, [i for i, n in enumerate(self.dec_names) if n.endswith("sin")]].sum())
        parts["Ngày lễ"] = float(attr_d[:, [self.dec_names.index("is_holiday"), self.dec_names.index("is_xmas")]].sum())
        start = origin + self.history
        return {"heatmap": attr_e[-72:].T.round(2).tolist(), "features": self.enc_names,
                "contributions": sorted(({"name": k, "mw": round(v, 1)} for k, v in parts.items()), key=lambda r: -abs(r["mw"])),
                "holiday_hours": int(self.holiday[start:start + self.horizon].sum())}

    def status(self) -> dict:
        if not self.ready.is_set():
            return {"ready": False, "message": "Đang nạp checkpoint và tạo dự báo test..."}
        if self.error:
            return {"ready": False, "error": self.error}
        return {"ready": True, "origins": len(self.models["final"]["mean"]),
                "models": {k: {"label": v["label"], "c": v["c"]} for k, v in self.models.items()}}

    def summary(self) -> list:
        return [row for key in ("midterm", "final") for row in self.models[key]["summary"]]

    def forecast(self, origin: int, model_key: str = "final") -> dict:
        if not self.ready.is_set() or self.error:
            raise RuntimeError("Forecast data is not ready.")
        m = self.models.get(model_key, self.models["final"])
        origin = max(0, min(origin, len(m["mean"]) - 1))
        history, horizon, frame = self.history, self.horizon, self.test_frame
        actual, mean, sigma = m["actual"][origin], m["mean"][origin], m["sigma"][origin]
        original_lower, original_upper = mean - Z_975 * sigma, mean + Z_975 * sigma
        calibrated_lower, calibrated_upper = mean - Z_975 * m["c"] * sigma, mean + Z_975 * m["c"] * sigma

        def interval_metrics(lower, upper):
            return {"picp": float(100 * np.mean((actual > lower) & (actual <= upper))), "sharpness": float(np.mean(upper - lower))}

        return {
            "model": model_key if model_key in self.models else "final", "label": m["label"],
            "explain": self.explain(origin) if model_key != "midterm" else None,
            "origin": origin, "origin_count": len(m["mean"]), "c": m["c"],
            "history": {"timestamps": [t.isoformat() for t in frame.index[origin:origin + history]], "load_mw": frame[TARGET].iloc[origin:origin + history].round(3).tolist()},
            "forecast": {
                "timestamps": [t.isoformat() for t in frame.index[origin + history:origin + history + horizon]],
                "actual_mw": actual.round(3).tolist(), "mean_mw": mean.round(3).tolist(),
                "original_lower_mw": original_lower.round(3).tolist(), "original_upper_mw": original_upper.round(3).tolist(),
                "calibrated_lower_mw": calibrated_lower.round(3).tolist(), "calibrated_upper_mw": calibrated_upper.round(3).tolist(),
            },
            "metrics": {"rmse": float(np.sqrt(np.mean((actual - mean) ** 2))), "mae": float(np.mean(np.abs(actual - mean))), "original": interval_metrics(original_lower, original_upper), "calibrated": interval_metrics(calibrated_lower, calibrated_upper)},
        }


STORE = ForecastStore()


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        return

    def send_json(self, payload, status=HTTPStatus.OK):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def serve_static(self, filename, content_type):
        path = STATIC_DIR / filename
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_json(STORE.status())
        elif parsed.path == "/api/summary":
            status = STORE.status(); self.send_json({**status, "summary": STORE.summary() if status.get("ready") else []})
        elif parsed.path == "/api/forecast":
            try:
                query = parse_qs(parsed.query)
                origin = int(query.get("origin", [1000])[0])
                self.send_json(STORE.forecast(origin, query.get("model", ["final"])[0]))
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        elif parsed.path in ("/", "/index.html"):
            self.serve_static("index.html", "text/html; charset=utf-8")
        elif parsed.path == "/styles.css":
            self.serve_static("styles.css", "text/css; charset=utf-8")
        elif parsed.path == "/app.js":
            self.serve_static("app.js", "application/javascript; charset=utf-8")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    STORE.start_loading()
    server = ThreadingHTTPServer(("127.0.0.1", 8501), RequestHandler)
    print("ProLoaF HTML demo: http://127.0.0.1:8501")
    print("Loading the local checkpoint in the background...")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
