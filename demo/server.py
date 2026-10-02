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
            self.data = {
                "test_frame": test_frame,
                "actual": targets[:, :, 0].detach().cpu().numpy() * load_range + load_min,
                "mean": raw_prediction[:, :, 0].detach().cpu().numpy() * load_range + load_min,
                "sigma": torch.exp(0.5 * raw_prediction[:, :, 1]).detach().cpu().numpy() * load_range,
                "c": float(calibration["selected_c"]), "history": int(config["history_horizon"]), "horizon": int(config["forecast_horizon"]),
                "summary": pd.read_csv(TEST_SUMMARY_PATH).to_dict(orient="records"),
            }
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.ready.set()

    def status(self) -> dict:
        if not self.ready.is_set():
            return {"ready": False, "message": "Đang nạp checkpoint và tạo dự báo test..."}
        if self.error:
            return {"ready": False, "error": self.error}
        return {"ready": True, "origins": len(self.data["mean"]), "c": self.data["c"]}

    def forecast(self, origin: int) -> dict:
        if not self.ready.is_set() or self.error:
            raise RuntimeError("Forecast data is not ready.")
        origin = max(0, min(origin, len(self.data["mean"]) - 1))
        history, horizon, frame = self.data["history"], self.data["horizon"], self.data["test_frame"]
        actual, mean, sigma = self.data["actual"][origin], self.data["mean"][origin], self.data["sigma"][origin]
        original_lower, original_upper = mean - Z_975 * sigma, mean + Z_975 * sigma
        calibrated_lower, calibrated_upper = mean - Z_975 * self.data["c"] * sigma, mean + Z_975 * self.data["c"] * sigma

        def interval_metrics(lower, upper):
            return {"picp": float(100 * np.mean((actual > lower) & (actual <= upper))), "sharpness": float(np.mean(upper - lower))}

        return {
            "origin": origin, "origin_count": len(self.data["mean"]), "c": self.data["c"],
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
            status = STORE.status(); self.send_json({**status, "summary": STORE.data["summary"] if status.get("ready") else []})
        elif parsed.path == "/api/forecast":
            try:
                origin = int(parse_qs(parsed.query).get("origin", [1000])[0]); self.send_json(STORE.forecast(origin))
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
