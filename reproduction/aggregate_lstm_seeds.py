"""Aggregate strict 60/20/20 LSTM runs into one summary.

Reads only finished run folders; writes a per-seed table, mean +/- sample std,
and RMSE by horizon averaged over seeds. No model is trained or changed.
Default seeds 1-3 rebuild `strict_602020_3seeds_summary`; `--seeds 1,2,3,4,5`
writes `strict_602020_5seeds_summary`.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/paper_reproduction'
METRICS = ['RMSE_MW', 'MAE_MW', 'PICP_percent', 'Sharpness_MW', 'MIS_standard_95_MW']


def run_folder(seed):
    # Seed 1 was the first strict run and kept its dated folder name.
    return 'strict_602020_20261013' if seed == 1 else f'strict_602020_seed{seed}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', default='1,2,3', help='comma-separated seeds to aggregate.')
    seeds = [int(s) for s in parser.parse_args().seeds.split(',')]
    out = ART / f'strict_602020_{len(seeds)}seeds_summary'
    rows, horizons = [], []
    for seed in seeds:
        folder = run_folder(seed)
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
    out.mkdir(exist_ok=True)
    per_seed.to_csv(out / 'per_seed.csv', index=False)
    by_horizon.to_csv(out / f'rmse_by_horizon_mean{len(seeds)}.csv', index=False)
    (out / 'aggregate.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print(per_seed.round(2).to_string(index=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
