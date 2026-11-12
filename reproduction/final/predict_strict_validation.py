"""Save B (validation) predictions of the five finished strict LSTM checkpoints.

No training. B predictions are needed to fit ensemble/conformal settings without touching C.
Output: artifacts/final_exam/runs/baseline_seed{s}/validation_predictions.npz and a copy of
the already saved C predictions, so all variants share one folder layout.
"""
import json
import shutil
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper_source/src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proloaf.datahandler as dh  # noqa: E402
import proloaf.modelhandler as mh  # noqa: E402
import proloaf.tensorloader as tl  # noqa: E402
from run_lstm_variant import predict  # noqa: E402

ART = ROOT / 'artifacts/paper_reproduction'
FOLDERS = {1: 'strict_602020_20261013', 2: 'strict_602020_seed2', 3: 'strict_602020_seed3',
           4: 'strict_602020_seed4', 5: 'strict_602020_seed5'}


def main():
    torch.set_num_threads(1)
    config = json.loads((ART / 'training_config_used.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True)
    train, validation, _ = dh.split(frame, [0.6, 0.8])
    scaler = dh.MultiScaler(config['feature_groups'])

    def dataset(df, fit=False):
        return tl.TimeSeriesData(df.copy(), device='cpu', preparation_steps=[
            dh.set_to_hours, partial(dh.fill_if_missing, periodicity=24),
            dh.add_cyclical_features, dh.add_onehot_features,
            scaler.fit_transform if fit else scaler.transform, dh.check_continuity], **config)

    dataset(train, fit=True).to_tensor()  # fits the scaler on A only
    validation_data = dataset(validation).to_tensor()
    for seed, folder in FOLDERS.items():
        out = ROOT / 'artifacts/final_exam/runs' / f'baseline_seed{seed}'
        out.mkdir(parents=True, exist_ok=True)
        model = mh.ModelHandler.load_model(str(ART / folder / 'opsd_recurrent.pkl'), locate='cpu')
        model.to('cpu')
        model.model.eval()
        y, m, s = predict(model, validation_data)
        np.savez_compressed(out / 'validation_predictions.npz', actual_scaled=y, mean_scaled=m, sigma_scaled=s)
        shutil.copy(ART / folder / 'test_predictions.npz', out / 'test_predictions.npz')
        print(seed, 'B RMSE MW', float(np.sqrt(np.mean((y - m) ** 2)) * 46242))


if __name__ == '__main__':
    main()
