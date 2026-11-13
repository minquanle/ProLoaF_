"""Evaluate final-exam improvements from saved predictions only (no training).

For each variant (baseline = midterm LSTM-chuan, at, at_hol) with seeds 1-5:
  * single seeds: point and 95% interval metrics on C, mean +/- std over seeds;
  * ensemble: mu = mean of seed means, sigma from the Gaussian mixture of the 5 seeds;
  * interval calibration fitted on B only, applied once to C:
      global  : one c for all horizons (midterm method),
      per_h   : one c_h per forecast hour (split conformal per horizon).
Statistical tests:
  * Welch t-test of single-seed RMSE versus DLinear (n = 5 each);
  * Diebold-Mariano style paired t-test of daily squared error, ensemble versus
    DLinear ensemble, on non-overlapping forecast origins (one per 24 h).
Holiday error: MAE on German national holidays versus other hours of C.
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

RUNS = ROOT / 'artifacts/final_exam/runs'
OUT = ROOT / 'artifacts/final_exam/results'
SCALE = 46242.0
Z = 1.959963984540054
SEEDS = [1, 2, 3, 4, 5]
DLINEAR = {s: ROOT / 'artifacts/paper_reproduction' / ('dlinear_opsd_20261015' if s <= 3 else 'dlinear_opsd_seed45')
           / f'test_predictions_seed{s}.npz' for s in SEEDS}


def interval_metrics(y, m, s, c):
    lo, hi = m - Z * c * s, m + Z * c * s
    width = hi - lo
    mis = width + 40.0 * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return {'RMSE_MW': float(np.sqrt(np.mean((y - m) ** 2)) * SCALE), 'MAE_MW': float(np.mean(np.abs(y - m)) * SCALE),
            'PICP_percent': float(np.mean((y >= lo) & (y <= hi)) * 100), 'Width_MW': float(width.mean() * SCALE),
            'MIS_MW': float(mis.mean() * SCALE)}


def conformal_c(y, m, s, axis=None):
    r = np.abs(y - m) / (Z * s)
    n = r.shape[0] if axis == 0 else r.size
    level = min(1.0, np.ceil((n + 1) * 0.95) / n)
    return np.quantile(r, level, axis=axis, method='higher')


def ensemble(preds):
    mu = np.mean([p['mean_scaled'] for p in preds], axis=0)
    second = np.mean([p['sigma_scaled'] ** 2 + p['mean_scaled'] ** 2 for p in preds], axis=0)
    return mu, np.sqrt(np.maximum(second - mu ** 2, 1e-12))


def load(variant, split):
    files = [RUNS / f'{variant}_seed{s}' / f'{split}_predictions.npz' for s in SEEDS]
    if not all(f.exists() for f in files):
        return None
    return [dict(np.load(f)) for f in files]


def holiday_mask():
    frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True)
    flags = add_calendar_flags(frame[[]])
    test = flags.iloc[int(len(flags) * 0.8):]
    idx = np.arange(8595)[:, None] + 147 + np.arange(24)[None, :]
    return test['is_holiday'].to_numpy()[idx] > 0


def daily_dm(err_a, err_b):
    """Paired t-test of daily MSE difference on non-overlapping origins (DM with h=1 days)."""
    d = (err_a[::24] ** 2).mean(axis=1) - (err_b[::24] ** 2).mean(axis=1)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return {'days': int(len(d)), 'mean_loss_diff_MW2': float(d.mean() * SCALE ** 2),
            'dm_t': float(t), 'p_two_sided': float(2 * stats.t.sf(abs(t), len(d) - 1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    holiday = holiday_mask()
    dl = [dict(np.load(DLINEAR[s])) for s in SEEDS]
    y = dl[0]['actual_scaled']
    dl_rmse = [float(np.sqrt(np.mean((d['actual_scaled'] - d['mean_scaled']) ** 2)) * SCALE) for d in dl]
    dl_ens = np.mean([d['mean_scaled'] for d in dl], axis=0)
    rows, tests, horizon_rows = [], {}, []
    rows.append({'variant': 'DLinear', 'model': 'single seeds (mean)', 'RMSE_MW': np.mean(dl_rmse),
                 'RMSE_std_MW': np.std(dl_rmse, ddof=1)})
    rows.append({'variant': 'DLinear', 'model': 'ensemble 5 seeds',
                 **{k: v for k, v in interval_metrics(y, dl_ens, np.ones_like(y), 0).items() if k in ('RMSE_MW', 'MAE_MW')}})
    for variant in ['baseline', 'at', 'at_hol']:
        test, val = load(variant, 'test'), load(variant, 'validation')
        if test is None or val is None:
            print('skip', variant, '(runs not finished)')
            continue
        assert np.allclose(test[0]['actual_scaled'], y, atol=1e-6)
        singles = [interval_metrics(p['actual_scaled'], p['mean_scaled'], p['sigma_scaled'], 1.0) for p in test]
        single = pd.DataFrame(singles)
        rmse = single['RMSE_MW'].to_numpy()
        rows.append({'variant': variant, 'model': 'single seeds (mean)', **single.mean().to_dict(),
                     'RMSE_std_MW': float(rmse.std(ddof=1))})
        welch = stats.ttest_ind(rmse, dl_rmse, equal_var=False)
        mu_c, sd_c = ensemble(test)
        mu_b, sd_b = ensemble(val)
        yb = val[0]['actual_scaled']
        c_global = float(conformal_c(yb, mu_b, sd_b))
        c_h = conformal_c(yb, mu_b, sd_b, axis=0)
        for name, c in [('ensemble raw', 1.0), ('ensemble + global c', c_global), ('ensemble + per-hour c', c_h[None, :])]:
            m = interval_metrics(y, mu_c, sd_c, c)
            err = np.abs(y - mu_c) * SCALE
            rows.append({'variant': variant, 'model': name, **m,
                         'MAE_holiday_MW': float(err[holiday].mean()), 'MAE_other_MW': float(err[~holiday].mean())})
        for h in range(24):
            horizon_rows.append({'variant': variant, 'horizon': h + 1, 'c_h': float(c_h[h]),
                                 **interval_metrics(y[:, h], mu_c[:, h], sd_c[:, h], c_h[h])})
        tests[variant] = {
            'single_seed_RMSE_vs_DLinear_welch_t': float(welch.statistic),
            'single_seed_RMSE_vs_DLinear_welch_p': float(welch.pvalue),
            'ensemble_vs_DLinear_ensemble_DM': daily_dm(y - mu_c, y - dl_ens),
            'c_global_from_B': c_global, 'c_per_hour_from_B': [float(v) for v in c_h]}
        np.savez_compressed(OUT / f'{variant}_ensemble_test.npz', actual_scaled=y, mean_scaled=mu_c, sigma_scaled=sd_c,
                            c_per_hour=c_h)
    if 'baseline' in tests:
        for variant in ['at', 'at_hol']:
            if variant in tests:
                a = np.load(OUT / f'{variant}_ensemble_test.npz')['mean_scaled']
                b = np.load(OUT / 'baseline_ensemble_test.npz')['mean_scaled']
                tests[variant]['ensemble_vs_baseline_ensemble_DM'] = daily_dm(y - a, y - b)
    dlc_dir = ROOT / 'artifacts/final_exam/dlinear_calendar'
    if (dlc_dir / 'test_summary_per_seed.csv').exists() and 'at_hol' in tests:
        dlc = [dict(np.load(dlc_dir / f'test_predictions_seed{s}.npz')) for s in SEEDS]
        dlc_rmse = [float(np.sqrt(np.mean((d['actual_scaled'] - d['mean_scaled']) ** 2)) * SCALE) for d in dlc]
        dlc_ens = np.mean([d['mean_scaled'] for d in dlc], axis=0)
        err = np.abs(y - dlc_ens) * SCALE
        rows.append({'variant': 'DLinear+calendar', 'model': 'single seeds (mean)', 'RMSE_MW': np.mean(dlc_rmse),
                     'RMSE_std_MW': np.std(dlc_rmse, ddof=1)})
        rows.append({'variant': 'DLinear+calendar', 'model': 'ensemble 5 seeds',
                     'RMSE_MW': float(np.sqrt(np.mean((y - dlc_ens) ** 2)) * SCALE), 'MAE_MW': float(err.mean()),
                     'MAE_holiday_MW': float(err[holiday].mean()), 'MAE_other_MW': float(err[~holiday].mean())})
        a = np.load(OUT / 'at_hol_ensemble_test.npz')['mean_scaled']
        rmse_hol = [interval_metrics(p['actual_scaled'], p['mean_scaled'], p['sigma_scaled'], 1)['RMSE_MW'] for p in load('at_hol', 'test')]
        tests['at_hol']['ensemble_vs_DLinear_calendar_ensemble_DM'] = daily_dm(y - a, y - dlc_ens)
        tests['at_hol']['single_seed_RMSE_vs_DLinear_calendar_welch_p'] = float(
            stats.ttest_ind(rmse_hol, dlc_rmse, equal_var=False).pvalue)
    if 'at' in tests and 'at_hol' in tests:
        a = np.load(OUT / 'at_hol_ensemble_test.npz')['mean_scaled']
        b = np.load(OUT / 'at_ensemble_test.npz')['mean_scaled']
        tests['at_hol']['ensemble_vs_at_ensemble_DM'] = daily_dm(y - a, y - b)
        rmse_at = [interval_metrics(p['actual_scaled'], p['mean_scaled'], p['sigma_scaled'], 1)['RMSE_MW'] for p in load('at', 'test')]
        rmse_hol = [interval_metrics(p['actual_scaled'], p['mean_scaled'], p['sigma_scaled'], 1)['RMSE_MW'] for p in load('at_hol', 'test')]
        welch = stats.ttest_ind(rmse_hol, rmse_at, equal_var=False)
        tests['at_hol']['single_seed_RMSE_vs_at_welch_p'] = float(welch.pvalue)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(horizon_rows).to_csv(OUT / 'per_horizon.csv', index=False)
    (OUT / 'tests.json').write_text(json.dumps(tests, indent=2), encoding='utf-8')
    pd.set_option('display.width', 200)
    print(table.round(2).to_string(index=False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'c_per_hour_from_B'} for k, v in tests.items()}, indent=1))


if __name__ == '__main__':
    main()
