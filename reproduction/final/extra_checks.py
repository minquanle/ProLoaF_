"""Extra robustness checks for the final report, from saved predictions only (no training).

1. Conditional coverage of the calibrated 95% interval on C (2019): by season, day type and
   forecast-hour block, for the three LSTM variants (ensemble + global c chosen on B).
2. Diebold-Mariano tests with a Newey-West (HAC) variance on the daily loss differences,
   next to the plain paired t-test used in evaluate_improvements.py.

Output: artifacts/final_exam/results/coverage_breakdown.csv, dm_robust.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_lstm_variant import add_calendar_flags  # noqa: E402

OUT = ROOT / 'artifacts/final_exam/results'
SCALE, Z, SEEDS = 46242.0, 1.959963984540054, [1, 2, 3, 4, 5]
HAC_LAGS = 7


def groups():
    frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True)
    flags = add_calendar_flags(frame[[]])
    test = flags.iloc[int(len(flags) * 0.8):]
    idx = np.arange(8595)[:, None] + 147 + np.arange(24)[None, :]
    local = test.index.tz_localize('UTC').tz_convert('Europe/Berlin')
    month, dow = local.month.to_numpy()[idx], local.dayofweek.to_numpy()[idx]
    hol = test['is_holiday'].to_numpy()[idx] > 0
    xmas = (test['is_xmas'].to_numpy()[idx] > 0) & ~hol
    horizon = np.broadcast_to(np.arange(1, 25), idx.shape)
    special = hol | xmas
    return {
        'Cả năm': np.ones(idx.shape, bool),
        'Mùa đông (12, 1, 2)': np.isin(month, [12, 1, 2]),
        'Mùa xuân (3-5)': np.isin(month, [3, 4, 5]),
        'Mùa hè (6-8)': np.isin(month, [6, 7, 8]),
        'Mùa thu (9-11)': np.isin(month, [9, 10, 11]),
        'Ngày làm việc (không lễ)': (dow < 5) & ~special,
        'Cuối tuần (không lễ)': (dow >= 5) & ~special,
        'Ngày lễ toàn quốc': hol,
        'Giáng sinh - năm mới (không lễ)': xmas,
        'Giờ dự báo 1-6': horizon <= 6,
        'Giờ dự báo 7-12': (horizon > 6) & (horizon <= 12),
        'Giờ dự báo 13-24': horizon > 12,
    }


def dm(err_a, err_b, lags=HAC_LAGS):
    d = (err_a[::24] ** 2).mean(axis=1) - (err_b[::24] ** 2).mean(axis=1)
    n, mean = len(d), d.mean()
    plain = mean / (d.std(ddof=1) / np.sqrt(n))
    e = d - mean
    var = (e @ e) / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    hac = mean / np.sqrt(var / n)
    return {'days': int(n), 't_plain': float(plain), 'p_plain': float(2 * stats.t.sf(abs(plain), n - 1)),
            't_hac': float(hac), 'p_hac': float(2 * stats.norm.sf(abs(hac))), 'hac_lags_days': lags}


def main():
    tests = json.loads((OUT / 'tests.json').read_text(encoding='utf-8'))
    masks = groups()
    preds = {v: np.load(OUT / f'{v}_ensemble_test.npz') for v in ['baseline', 'at', 'at_hol']}
    y = preds['at_hol']['actual_scaled']
    rows = []
    for name, mask in masks.items():
        row = {'group': name, 'share_percent': float(mask.mean() * 100)}
        for v, p in preds.items():
            c = tests[v]['c_global_from_B']
            half = Z * c * p['sigma_scaled']
            inside = np.abs(y - p['mean_scaled']) <= half
            row[f'{v}_PICP'] = float(inside[mask].mean() * 100)
            row[f'{v}_width_MW'] = float((2 * half)[mask].mean() * SCALE)
            row[f'{v}_MAE_MW'] = float(np.abs(y - p['mean_scaled'])[mask].mean() * SCALE)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'coverage_breakdown.csv', index=False)
    pd.set_option('display.width', 250)
    print(table.round(2).to_string(index=False))

    pr = ROOT / 'artifacts/paper_reproduction'
    dl = np.mean([np.load(pr / ('dlinear_opsd_20261015' if s <= 3 else 'dlinear_opsd_seed45') / f'test_predictions_seed{s}.npz')['mean_scaled']
                  for s in SEEDS], axis=0)
    dlc = np.mean([np.load(ROOT / f'artifacts/final_exam/dlinear_calendar/test_predictions_seed{s}.npz')['mean_scaled']
                   for s in SEEDS], axis=0)
    m = {v: p['mean_scaled'] for v, p in preds.items()}
    pairs = {'at vs baseline (C1)': (m['at'], m['baseline']), 'at_hol vs at (C2)': (m['at_hol'], m['at']),
             'at_hol vs baseline': (m['at_hol'], m['baseline']), 'at_hol vs DLinear': (m['at_hol'], dl),
             'at_hol vs DLinear+calendar': (m['at_hol'], dlc), 'baseline vs DLinear': (m['baseline'], dl)}
    robust = {k: dm(y - a, y - b) for k, (a, b) in pairs.items()}
    (OUT / 'dm_robust.json').write_text(json.dumps(robust, indent=2), encoding='utf-8')
    for k, v in robust.items():
        print(f"{k:28s} plain t={v['t_plain']:7.2f} p={v['p_plain']:.2e} | HAC t={v['t_hac']:7.2f} p={v['p_hac']:.2e}")


if __name__ == '__main__':
    main()
