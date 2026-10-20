"""Aggregate the three strict 60/20/20 LSTM runs (seeds 1, 2, 3) into one summary.

Reads only finished run folders; writes a per-seed table, mean +/- sample std,
and RMSE by horizon averaged over seeds. No model is trained or changed.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/paper_reproduction'
RUNS = {1: 'strict_602020_20261013', 2: 'strict_602020_seed2', 3: 'strict_602020_seed3'}
OUT = ART / 'strict_602020_3seeds_summary'
METRICS = ['RMSE_MW', 'MAE_MW', 'PICP_percent', 'Sharpness_MW', 'MIS_standard_95_MW']


def main():
    rows, horizons = [], []
    for seed, folder in RUNS.items():
        status = json.loads((ART / folder / 'status.json').read_text(encoding='utf-8'))
        assert status['state'] == 'complete', folder
        protocol = json.loads((ART / folder / 'protocol.json').read_text(encoding='utf-8'))
        assert protocol['seed'] == seed, folder
        summary = status['test_summary']
        rows.append({'seed': seed, 'folder': folder, 'best_epoch': status['best_epoch'],
                     'stopped_epoch': status['actual_epochs'], **{m: summary[m] for m in METRICS}})
        by_h = pd.read_csv(ART / folder / 'test_metrics_by_horizon.csv')
        scale = protocol['target_range_MW']
        horizons.append(pd.DataFrame({'seed': seed, 'horizon': by_h['horizon_hour'], 'RMSE_MW': by_h['RMSE'] * scale}))
    per_seed = pd.DataFrame(rows)
    aggregate = {m: {'mean': float(per_seed[m].mean()), 'sample_std': float(per_seed[m].std(ddof=1))} for m in METRICS}
    by_horizon = pd.concat(horizons).groupby('horizon', as_index=False)['RMSE_MW'].mean()
    OUT.mkdir(exist_ok=True)
    per_seed.to_csv(OUT / 'per_seed.csv', index=False)
    by_horizon.to_csv(OUT / 'rmse_by_horizon_mean3.csv', index=False)
    (OUT / 'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print(per_seed.round(2).to_string(index=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
