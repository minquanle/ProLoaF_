"""DLinear with the same known-future calendar as the final LSTM (fairer comparison).

Model: official DLinear (pinned LTSF-Linear commit) on 147 h of DE load, plus a linear
term on the 48 future calendar columns of each forecast hour (24 hour, 12 month,
7 weekday one-hot, 3 sin, is_holiday, is_xmas), shared over the 24 hours:
    y_hat[h] = DLinear(x)[h] + w . cal[h] + b
Training protocol identical to the midterm DLinear: A train, B early stopping (MSE,
patience 7, max 50 epochs), Adam lr 1e-3, batch 32, seeds 1-5. C is predicted once.
Output: artifacts/final_exam/dlinear_calendar/

With --per-horizon the calendar term gets its own weights for each forecast hour,
    y_hat[h] = DLinear(x)[h] + w_h . cal[h] + b_h
so a holiday can change the shape of the day (output: dlinear_calendar_perh/).
"""
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper_source/src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proloaf.datahandler as dh  # noqa: E402
from run_lstm_variant import add_calendar_flags  # noqa: E402

PER_HORIZON = '--per-horizon' in sys.argv
OUT = ROOT / 'artifacts/final_exam' / ('dlinear_calendar_perh' if PER_HORIZON else 'dlinear_calendar')
TARGET = 'DE_load_actual_entsoe_transparency'


class DLinearCalendar(torch.nn.Module):
    def __init__(self, module, n_cal):
        super().__init__()
        self.base = module.Model(SimpleNamespace(seq_len=147, pred_len=24, enc_in=1, individual=False))
        self.per_horizon = PER_HORIZON
        if PER_HORIZON:
            self.weight = torch.nn.Parameter(torch.zeros(24, n_cal))
            self.bias = torch.nn.Parameter(torch.zeros(24))
        else:
            self.cal = torch.nn.Linear(n_cal, 1)

    def forward(self, x, cal):
        if self.per_horizon:
            return self.base(x) + ((cal * self.weight).sum(-1) + self.bias).unsqueeze(-1)
        return self.base(x) + self.cal(cal)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('official_dlinear', ROOT / 'external/LTSF-Linear/models/DLinear.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cal_cols = json.loads((ROOT / 'artifacts/final_exam/runs/at_hol_seed1/training_config_used.json')
                          .read_text(encoding='utf-8'))['decoder_features']
    frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True)
    frame = dh.add_onehot_features(dh.add_cyclical_features(add_calendar_flags(frame)))
    load = frame[TARGET].to_numpy(np.float64)
    a, b = int(len(frame) * 0.6), int(len(frame) * 0.8)
    minimum, scale = float(load[:a].min()), float(np.ptp(load[:a]))
    scaled = torch.tensor((load - minimum) / scale, dtype=torch.float32)
    cal = torch.tensor(frame[cal_cols].to_numpy(np.float32))

    def windows(lo, hi):
        y = scaled[lo:hi].unfold(0, 171, 1)
        c = cal[lo:hi].unfold(0, 171, 1)[:, :, 147:].transpose(1, 2)  # (n, 24, n_cal)
        return TensorDataset(y[:, :147].unsqueeze(-1), c, y[:, 147:].unsqueeze(-1))

    train, val, test = windows(0, a), windows(a, b), windows(b, len(frame))
    assert (len(train), len(val), len(test)) == (26124, 8595, 8595)
    rows, history = [], []
    for seed in [1, 2, 3, 4, 5]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = DLinearCalendar(module, len(cal_cols))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, stale, best_epoch, start = float('inf'), 0, 0, time.perf_counter()
        for epoch in range(1, 51):
            model.train()
            for x, c, y in DataLoader(train, batch_size=32, shuffle=True):
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(x, c), y)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(torch.nn.functional.mse_loss(model(*val.tensors[:2]), val.tensors[2]))
            history.append({'seed': seed, 'epoch': epoch, 'validation_B_mse': val_loss})
            if val_loss < best:
                best, stale, best_epoch = val_loss, 0, epoch
                torch.save(model.state_dict(), OUT / f'dlinear_calendar_seed{seed}.pt')
            else:
                stale += 1
                if stale >= 7:
                    break
        model.load_state_dict(torch.load(OUT / f'dlinear_calendar_seed{seed}.pt'))
        model.eval()
        with torch.no_grad():
            pred = model(*test.tensors[:2])[:, :, 0].numpy()
        actual = test.tensors[2][:, :, 0].numpy()
        np.savez_compressed(OUT / f'test_predictions_seed{seed}.npz', actual_scaled=actual, mean_scaled=pred)
        err = (actual - pred) * scale
        rows.append({'seed': seed, 'best_epoch': best_epoch, 'epochs': epoch, 'best_validation_mse': best,
                     'RMSE_MW': float(np.sqrt(np.mean(err ** 2))), 'MAE_MW': float(np.mean(np.abs(err))),
                     'train_seconds': time.perf_counter() - start,
                     'parameter_count': sum(p.numel() for p in model.parameters())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'test_summary_per_seed.csv', index=False)
    pd.DataFrame(history).to_csv(OUT / 'epoch_history.csv', index=False)
    (OUT / 'protocol.json').write_text(json.dumps({'calendar_columns': cal_cols, 'target_min_MW': minimum,
                                                   'target_range_MW': scale, 'selection': 'B MSE only'}, indent=2),
                                       encoding='utf-8')


if __name__ == '__main__':
    main()
