"""Diagnosis of the midterm strict LSTM on B (2018) only, so design decisions do not use C.

1. Error on German national holidays versus other hours (5-seed ensemble, saved B predictions).
2. Permutation importance on B (5-seed ensemble).
Output: artifacts/final_exam/diagnosis_B/. No training.
"""
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / 'paper_source/src'))
import proloaf.datahandler as dh  # noqa: E402
import proloaf.tensorloader as tl  # noqa: E402
import xai_analysis as xa  # noqa: E402  (baseline mode: no argv given)
from run_lstm_variant import add_calendar_flags  # noqa: E402

OUT = ROOT / 'artifacts/final_exam/diagnosis_B'
SCALE = 46242.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True)
    train, validation, _ = dh.split(frame, [0.6, 0.8])

    # 1. holiday error on B from saved ensemble predictions
    preds = [np.load(ROOT / f'artifacts/final_exam/runs/baseline_seed{s}/validation_predictions.npz') for s in range(1, 6)]
    y = preds[0]['actual_scaled']
    mu = np.mean([p['mean_scaled'] for p in preds], axis=0)
    flags = add_calendar_flags(validation[[]])
    idx = np.arange(len(y))[:, None] + 147 + np.arange(24)[None, :]
    hol = flags['is_holiday'].to_numpy()[idx] > 0
    xmas = flags['is_xmas'].to_numpy()[idx] > 0
    err = np.abs(y - mu) * SCALE
    holiday = {'MAE_holiday_MW': float(err[hol].mean()), 'MAE_xmas_MW': float(err[xmas].mean()),
               'MAE_other_MW': float(err[~hol & ~xmas].mean()), 'holiday_hours_share': float(hol.mean())}
    (OUT / 'holiday_error_B.json').write_text(json.dumps(holiday, indent=2), encoding='utf-8')
    print(holiday, flush=True)

    # 2. permutation importance on B
    config = json.loads((xa.ART / 'training_config_used.json').read_text(encoding='utf-8'))
    scaler = dh.MultiScaler(config['feature_groups'])

    def dataset(df, fit=False):
        return tl.TimeSeriesData(df.copy(), device='cpu', preparation_steps=[
            dh.set_to_hours, partial(dh.fill_if_missing, periodicity=24),
            dh.add_cyclical_features, dh.add_onehot_features,
            scaler.fit_transform if fit else scaler.transform, dh.check_continuity], **config)

    dataset(train, fit=True).to_tensor()
    enc, dec, yt = next(iter(dataset(validation).to_tensor().make_data_loader(batch_size=10000, shuffle=False, drop_last=False)))
    nets = [xa.load_model(s) for s in xa.FOLDERS]
    base, perm = xa.permutation_importance(nets, enc, dec, yt[:, :, 0], config['encoder_features'], config['decoder_features'])
    perm.to_csv(OUT / 'permutation_importance_B.csv', index=False)
    print('B ensemble RMSE', round(base, 1))
    print(perm.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
