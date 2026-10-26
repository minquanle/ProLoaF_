"""Compare strict LSTM and DLinear across seeds 1-5 with simple significance tests.

Inputs are finished runs only: `strict_602020_5seeds_summary` (built by
aggregate_lstm_seeds.py --seeds 1,2,3,4,5) and the DLinear folders for seeds
1-3 and 4-5. Each seed gives one RMSE and one MAE on C, so each test has
n = 5 per model. The tests measure variation due to training randomness only,
not sampling variation of the test year. No model is trained or changed.
"""
import json
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/paper_reproduction'
LSTM_SUMMARY = ART / 'strict_602020_5seeds_summary'
DLINEAR_RUNS = [ART / 'dlinear_opsd_20261015', ART / 'dlinear_opsd_seed45']
OUT = ART / 'lstm_vs_dlinear_5seeds'
SEEDS = [1, 2, 3, 4, 5]


def main():
    lstm = pd.read_csv(LSTM_SUMMARY / 'per_seed.csv')
    dlinear = pd.concat(pd.read_csv(run / 'test_summary_per_seed.csv') for run in DLINEAR_RUNS)
    assert sorted(lstm['seed']) == SEEDS and sorted(dlinear['seed']) == SEEDS
    per_seed = pd.concat([
        lstm.assign(model='LSTM_strict')[['model', 'seed', 'RMSE_MW', 'MAE_MW']],
        dlinear.assign(model='DLinear')[['model', 'seed', 'RMSE_MW', 'MAE_MW']],
    ]).sort_values(['model', 'seed'])

    tests = {}
    for metric in ['RMSE_MW', 'MAE_MW']:
        a = lstm[metric].to_numpy()
        b = dlinear[metric].to_numpy()
        welch = stats.ttest_ind(a, b, equal_var=False)
        mann_whitney = stats.mannwhitneyu(a, b, alternative='two-sided', method='exact')
        tests[metric] = {
            'LSTM_mean': float(a.mean()), 'LSTM_sample_std': float(a.std(ddof=1)),
            'DLinear_mean': float(b.mean()), 'DLinear_sample_std': float(b.std(ddof=1)),
            'difference_LSTM_minus_DLinear': float(a.mean() - b.mean()),
            'relative_difference_percent': float(100 * (a.mean() - b.mean()) / b.mean()),
            'welch_t': float(welch.statistic), 'welch_p_two_sided': float(welch.pvalue),
            'mann_whitney_U': float(mann_whitney.statistic),
            'mann_whitney_p_two_sided': float(mann_whitney.pvalue),
        }

    dlinear_horizon = pd.concat(pd.read_csv(run / 'test_metrics_by_horizon.csv') for run in DLINEAR_RUNS)
    dlinear_horizon = dlinear_horizon.groupby('horizon', as_index=False)['RMSE_MW'].mean()
    lstm_horizon = pd.read_csv(LSTM_SUMMARY / 'rmse_by_horizon_mean5.csv')
    by_horizon = lstm_horizon.rename(columns={'RMSE_MW': 'LSTM_RMSE_MW'}).merge(
        dlinear_horizon.rename(columns={'RMSE_MW': 'DLinear_RMSE_MW'}), on='horizon')

    OUT.mkdir(exist_ok=True)
    per_seed.to_csv(OUT / 'per_seed.csv', index=False)
    by_horizon.to_csv(OUT / 'rmse_by_horizon_mean5.csv', index=False)
    (OUT / 'tests.json').write_text(json.dumps(tests, indent=2), encoding='utf-8')
    print(per_seed.round(2).to_string(index=False))
    print(json.dumps(tests, indent=2))
    print(by_horizon.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
