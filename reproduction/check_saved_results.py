"""Check every reported number from saved files only. No model is trained.

Run from the repository root:
    python reproduction/check_saved_results.py

1. LSTM-goc (author split): metrics from evaluation/opsd_recurrent.csv.
2. LSTM-chuan 60/20/20, seeds 1-5: RMSE/MAE recomputed from the saved
   test_predictions.npz and compared with the saved test_summary.csv.
3. DLinear seeds 1-5: same recomputation from test_predictions_seed*.npz.
4. Welch t-test LSTM vs DLinear recomputed from the per-seed RMSE.
5. Baselines, Prophet and interval calibration read from their CSV files.
The table is printed and written to artifacts/paper_reproduction/saved_results_check.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/paper_reproduction'
RANGE_MW = 46242.0  # min-max range of DE load on partition A
LSTM_FOLDERS = {1: 'strict_602020_20261013', 2: 'strict_602020_seed2', 3: 'strict_602020_seed3',
                4: 'strict_602020_seed4', 5: 'strict_602020_seed5'}
DLINEAR_FILES = {s: ART / 'dlinear_opsd_20261015' / f'test_predictions_seed{s}.npz' for s in (1, 2, 3)}
DLINEAR_FILES.update({s: ART / 'dlinear_opsd_seed45' / f'test_predictions_seed{s}.npz' for s in (4, 5)})


def rmse_mae_mw(npz_path):
    d = np.load(npz_path)
    err = (d['actual_scaled'] - d['mean_scaled']) * RANGE_MW
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def main():
    rows = []
    old = pd.read_csv(ART / 'evaluation/opsd_recurrent.csv', sep=';').iloc[0]
    rows.append(('LSTM-goc (tac gia, chia 0.6)', '-', old['Rmse'] * RANGE_MW, old['Mae'] * RANGE_MW, old['Picp'], 'saved'))

    lstm_rmse, dl_rmse = [], []
    for seed, folder in LSTM_FOLDERS.items():
        rmse, mae = rmse_mae_mw(ART / folder / 'test_predictions.npz')
        saved = pd.read_csv(ART / folder / 'test_summary.csv').iloc[0]
        ok = abs(rmse - saved['RMSE_MW']) < 0.5
        rows.append(('LSTM-chuan 60/20/20', seed, rmse, mae, saved['PICP_percent'], 'OK' if ok else 'MISMATCH'))
        lstm_rmse.append(rmse)

    dl_saved = pd.concat(pd.read_csv(p) for p in [ART / 'dlinear_opsd_20261015/test_summary_per_seed.csv',
                                                   ART / 'dlinear_opsd_seed45/test_summary_per_seed.csv'])
    for seed, path in DLINEAR_FILES.items():
        rmse, mae = rmse_mae_mw(path)
        saved = dl_saved.loc[dl_saved['seed'] == seed, 'RMSE_MW'].iloc[0]
        rows.append(('DLinear', seed, rmse, mae, np.nan, 'OK' if abs(rmse - saved) < 0.5 else 'MISMATCH'))
        dl_rmse.append(rmse)

    base = pd.read_csv(ART / 'baseline_comparison/two_baseline_comparison.csv')
    for _, r in base[base['model'] != 'ProLoaF-LSTM'].iterrows():
        rows.append((r['model'], '-', r['Rmse_MW'], r['Mae_MW'], r['Picp'], 'saved'))
    prophet = pd.read_csv(ART / 'prophet_paper_protocol/prophet_protocol_metrics.csv')
    r = prophet[prophet['protocol'] == 'chronological_aligned_95pct'].iloc[0]
    rows.append(('Prophet (canh thoi gian, 95%)', '-', r['Rmse_MW'], r['Mae_MW'], r['Picp'], 'saved'))
    cal = pd.read_csv(ART / 'interval_calibration/test_calibrated_summary.csv')
    r = cal[cal['interval'] == 'validation_calibrated_frozen_c'].iloc[0]
    rows.append((f"LSTM-goc hieu chinh c={r['c']:.4f}", '-', r['RMSE'] * RANGE_MW, r['MAE'] * RANGE_MW, r['PICP_percent'], 'saved'))

    table = pd.DataFrame(rows, columns=['model', 'seed', 'RMSE_MW', 'MAE_MW', 'PICP_percent', 'check'])
    welch = stats.ttest_ind(lstm_rmse, dl_rmse, equal_var=False)
    pd.set_option('display.width', 120)
    print(table.round(2).to_string(index=False))
    print(f"\nLSTM-chuan RMSE 5 seed: {np.mean(lstm_rmse):.1f} +/- {np.std(lstm_rmse, ddof=1):.1f} MW")
    print(f"DLinear    RMSE 5 seed: {np.mean(dl_rmse):.1f} +/- {np.std(dl_rmse, ddof=1):.1f} MW")
    print(f"Welch t = {welch.statistic:.3f}, p = {welch.pvalue:.3f}")
    table.to_csv(ART / 'saved_results_check.csv', index=False)
    if (table['check'] == 'MISMATCH').any():
        raise SystemExit('Some recomputed numbers do not match the saved files.')
    print('\nTat ca so lieu khop voi file da luu.')


if __name__ == '__main__':
    main()
