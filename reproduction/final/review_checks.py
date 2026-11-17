"""Checks added after the report review, from saved predictions only (no LSTM training).

1. Diebold-Mariano for each of the 24 possible daily origin offsets (not only offset 0),
   plus a sign test and a circular block bootstrap for the holiday-flag comparison (C2).
2. Relative point metrics (MAPE, nRMSE) and probabilistic metrics beyond the 95% level
   (Gaussian CRPS, pinball loss, coverage of the 50/80/90/95% central intervals).
3. Errors on the day after a national holiday and on bridge days.
4. Global c versus per-hour c judged on B only (fit on the first half of 2018, test on the second).
5. Forecast-hour 1: mean of single-seed RMSEs versus ensemble RMSE (LSTM and DLinear).

Output: artifacts/final_exam/results/review_checks.json and printed tables.
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

FE = ROOT / 'artifacts/final_exam'
OUT = FE / 'results'
PR = ROOT / 'artifacts/paper_reproduction'
SCALE, MINIMUM, Z, SEEDS = 46242.0, 31307.0, 1.959963984540054, [1, 2, 3, 4, 5]
rng = np.random.default_rng(0)


def dl_files(kind):
    if kind == 'DLinear':
        return [PR / ('dlinear_opsd_20261015' if s <= 3 else 'dlinear_opsd_seed45') / f'test_predictions_seed{s}.npz' for s in SEEDS]
    return [FE / kind / f'test_predictions_seed{s}.npz' for s in SEEDS]


def calendar():
    frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True)
    flags = add_calendar_flags(frame[[]])
    test = flags.iloc[int(len(flags) * 0.8):]
    idx = np.arange(8595)[:, None] + 147 + np.arange(24)[None, :]
    local = test.index.tz_localize('UTC').tz_convert('Europe/Berlin')
    day = pd.Series(local.date, index=test.index)
    hol_days = set(day[test['is_holiday'].to_numpy() > 0])
    dow = local.dayofweek.to_numpy()
    one = pd.Timedelta(days=1)
    after = np.array([(d - one) in hol_days and d not in hol_days and w < 5 for d, w in zip(day, dow)])
    bridge = np.array([d not in hol_days and ((w == 4 and (d - one) in hol_days) or (w == 0 and (d + one) in hol_days))
                       for d, w in zip(day, dow)])
    special = (test['is_holiday'].to_numpy() > 0) | (test['is_xmas'].to_numpy() > 0)
    normal_weekday = (dow < 5) & ~special & ~after
    return {'first_target_utc': str(test.index[147]), 'after_holiday': (after & ~special)[idx],
            'bridge': (bridge & ~special)[idx], 'normal_weekday': normal_weekday[idx],
            'bridge_days': sorted({str(d) for d, b, s in zip(day, bridge, special) if b and not s})}


def crps_gauss(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def main():
    tests = json.loads((OUT / 'tests.json').read_text(encoding='utf-8'))
    variants = [v for v in ['baseline', 'at', 'hol', 'noat', 'at_hol'] if (OUT / f'{v}_ensemble_test.npz').exists() and v in tests]
    preds = {v: np.load(OUT / f'{v}_ensemble_test.npz') for v in variants}
    y = preds['at_hol']['actual_scaled']
    mu = {v: p['mean_scaled'] for v, p in preds.items()}
    points = dict(mu)
    for kind in ['DLinear', 'dlinear_calendar', 'dlinear_calendar_perh']:
        if all(f.exists() for f in dl_files(kind)):
            points[kind] = np.mean([np.load(f)['mean_scaled'] for f in dl_files(kind)], axis=0)
    report = {}

    # 1. DM for every origin offset
    pairs = {'C1: at vs baseline': ('at', 'baseline'), 'C2: at_hol vs at': ('at_hol', 'at'),
             'at_hol vs baseline': ('at_hol', 'baseline'), 'at_hol vs DLinear': ('at_hol', 'DLinear'),
             'at_hol vs DLinear+calendar': ('at_hol', 'dlinear_calendar'),
             'at_hol vs DLinear+calendar per hour': ('at_hol', 'dlinear_calendar_perh'),
             'baseline vs DLinear': ('baseline', 'DLinear'),
             'C2 alone: hol vs baseline': ('hol', 'baseline'), 'C1 with C2: at_hol vs hol': ('at_hol', 'hol'),
             'noat vs baseline': ('noat', 'baseline'), 'at vs noat': ('at', 'noat')}
    cal = calendar()
    report['origin_offset_0_first_target_utc'] = cal['first_target_utc']
    report['dm_by_offset'] = {}
    print('DM over the 24 daily origin offsets (plain paired t):')
    for name, (a, b) in pairs.items():
        if a not in points or b not in points:
            continue
        ts, ps = [], []
        for off in range(24):
            d = ((y - points[a])[off::24] ** 2).mean(1) - ((y - points[b])[off::24] ** 2).mean(1)
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            ts.append(t)
            ps.append(2 * stats.t.sf(abs(t), len(d) - 1))
        report['dm_by_offset'][name] = {'t_min': float(min(ts)), 't_max': float(max(ts)), 'p_max': float(max(ps)),
                                        'offsets_p_below_0.05': int(np.sum(np.array(ps) < 0.05))}
        print(f"  {name:38s} t in [{min(ts):7.2f}, {max(ts):7.2f}]  largest p = {max(ps):.2e}  significant {np.sum(np.array(ps) < 0.05)}/24")

    # sign test and block bootstrap for the comparisons with small effects
    report['robust'] = {}
    for name in ['C2: at_hol vs at', 'C1: at vs baseline', 'baseline vs DLinear', 'C2 alone: hol vs baseline', 'at vs noat']:
        a, b = pairs[name]
        if a not in points or b not in points:
            continue
        d = ((y - points[a])[::24] ** 2).mean(1) - ((y - points[b])[::24] ** 2).mean(1)
        wins = int((d < 0).sum())
        sign_p = float(stats.binomtest(wins, len(d), 0.5).pvalue)
        wil_p = float(stats.wilcoxon(d).pvalue)
        block, n = 7, len(d)
        starts = rng.integers(0, n, size=(10000, int(np.ceil(n / block))))
        boot = np.array([d[(s[:, None] + np.arange(block)[None, :]).ravel()[:n] % n].mean() for s in starts])
        ci = np.percentile(boot, [2.5, 97.5]) * SCALE ** 2
        top10 = float(np.sort(d)[:10].sum() / d.sum())
        report['robust'][name] = {'days_first_model_better': wins, 'days': n, 'sign_test_p': sign_p, 'wilcoxon_p': wil_p,
                                  'block_bootstrap_95ci_MW2': [float(ci[0]), float(ci[1])],
                                  'bootstrap_share_above_zero': float((boot >= 0).mean()),
                                  'share_of_gain_from_10_best_days': top10}
        print(f"{name}: better on {wins}/{n} days, sign p={sign_p:.2e}, Wilcoxon p={wil_p:.2e}, "
              f"bootstrap 95% CI of mean loss diff [{ci[0]:.3g}, {ci[1]:.3g}] MW^2, top-10 days give {top10:.0%} of the gain")

    # 2. relative and probabilistic metrics
    actual = y * SCALE + MINIMUM
    rows = []
    for v, m in points.items():
        err = (y - m) * SCALE
        row = {'model': v, 'RMSE_MW': float(np.sqrt((err ** 2).mean())), 'MAE_MW': float(np.abs(err).mean()),
               'MAPE_percent': float((np.abs(err) / actual).mean() * 100),
               'nRMSE_percent_of_mean': float(np.sqrt((err ** 2).mean()) / actual.mean() * 100)}
        if v in preds:
            c = tests[v]['c_global_from_B']
            sd = preds[v]['sigma_scaled']
            row['CRPS_raw_MW'] = float(crps_gauss(y, m, sd).mean() * SCALE)
            row['CRPS_calibrated_MW'] = float(crps_gauss(y, m, c * sd).mean() * SCALE)
            for level in [50, 80, 90, 95]:
                z = stats.norm.ppf(0.5 + level / 200)
                row[f'cov{level}_raw'] = float((np.abs(y - m) <= z * sd).mean() * 100)
                row[f'cov{level}_cal'] = float((np.abs(y - m) <= z * c * sd).mean() * 100)
            q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
            pin = []
            for qq in q:
                pred = m + stats.norm.ppf(qq) * c * sd
                diff = y - pred
                pin.append(np.maximum(qq * diff, (qq - 1) * diff).mean())
            row['pinball_calibrated_MW'] = float(np.mean(pin) * SCALE)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'review_metrics.csv', index=False)
    pd.set_option('display.width', 260)
    print(table.round(2).to_string(index=False))

    # 3. day after a holiday and bridge days
    report['special_days'] = {'bridge_days_2019': cal['bridge_days']}
    for key in ['after_holiday', 'bridge', 'normal_weekday']:
        mask = cal[key]
        report['special_days'][key] = {'share_percent': float(mask.mean() * 100),
                                       **{v: float(np.abs((y - m)[mask]).mean() * SCALE) for v, m in points.items()}}
    print(json.dumps(report['special_days'], indent=1))

    # 4. global versus per-hour c, judged inside B
    report['c_choice_on_B'] = {}
    for v in variants:
        val = [np.load(FE / 'runs' / f'{v}_seed{s}' / 'validation_predictions.npz') for s in SEEDS]
        yb = val[0]['actual_scaled']
        mb = np.mean([p['mean_scaled'] for p in val], axis=0)
        sb = np.sqrt(np.maximum(np.mean([p['sigma_scaled'] ** 2 + p['mean_scaled'] ** 2 for p in val], axis=0) - mb ** 2, 1e-12))
        half = len(yb) // 2
        r = np.abs(yb - mb) / (Z * sb)
        cg = np.quantile(r[:half], 0.95)
        ch = np.quantile(r[:half], 0.95, axis=0)
        res = {}
        for name, c in [('global', cg), ('per_hour', ch[None, :])]:
            inside = r[half:] <= c
            res[name] = {'PICP': float(inside.mean() * 100), 'width_MW': float((2 * Z * c * sb[half:]).mean() * SCALE),
                         'PICP_by_hour_min': float(inside.mean(0).min() * 100), 'PICP_by_hour_max': float(inside.mean(0).max() * 100)}
        report['c_choice_on_B'][v] = res
        print(v, 'c fitted on first half of B, checked on second half:', json.dumps(res))

    # 5. forecast hour 1: single seeds versus ensemble
    report['hour1'] = {}
    singles = {'DLinear': [np.load(f) for f in dl_files('DLinear')]}
    for v in variants:
        singles[v] = [np.load(FE / 'runs' / f'{v}_seed{s}' / 'test_predictions.npz') for s in SEEDS]
    for v, runs in singles.items():
        per_seed = [float(np.sqrt(((p['actual_scaled'] - p['mean_scaled'])[:, 0] ** 2).mean()) * SCALE) for p in runs]
        ens = np.mean([p['mean_scaled'] for p in runs], axis=0)
        report['hour1'][v] = {'mean_of_seed_RMSE_MW': float(np.mean(per_seed)), 'seed_RMSE_MW': per_seed,
                              'ensemble_RMSE_MW': float(np.sqrt(((y - ens)[:, 0] ** 2).mean()) * SCALE)}
        print('h=1', v, report['hour1'][v])

    (OUT / 'review_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
