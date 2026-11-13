"""RMSE by forecast hour on 2019 (C) for the slides, from saved predictions only.

Output: artifacts/final_exam/figures/horizon_rmse.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / 'artifacts/final_exam'
SCALE = 46242.0
SEEDS = [1, 2, 3, 4, 5]
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})


def rmse_by_hour(y, mu):
    return np.sqrt(((y - mu) ** 2).mean(axis=0)) * SCALE


def ens(files):
    preds = [np.load(f) for f in files]
    return preds[0]['actual_scaled'], np.mean([p['mean_scaled'] for p in preds], axis=0)


def main():
    pr = ROOT / 'artifacts/paper_reproduction'
    dl_files = [pr / ('dlinear_opsd_20261015' if s <= 3 else 'dlinear_opsd_seed45') / f'test_predictions_seed{s}.npz'
                for s in SEEDS]
    dlc_files = [FE / 'dlinear_calendar' / f'test_predictions_seed{s}.npz' for s in SEEDS]
    hz = pd.read_csv(FE / 'results/per_horizon.csv')
    h = np.arange(1, 25)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(h, rmse_by_hour(*ens(dl_files)), '--', color='#9AA5B1', lw=2, label='DLinear')
    ax.plot(h, rmse_by_hour(*ens(dlc_files)), ':', color='#6B7785', lw=2.4, label='DLinear + lịch, ngày lễ')
    styles = {'baseline': ('LSTM-chuẩn (giữa kỳ)', '#E48E7E'), 'at': ('+ chuẩn hóa phụ tải Áo', '#8CC5DC'),
              'at_hol': ('+ cờ ngày lễ = LSTM-cuối kỳ', '#1F6F94')}
    for v, (label, color) in styles.items():
        ax.plot(h, hz[hz.variant == v].RMSE_MW.to_numpy(), '-o', ms=4, color=color, lw=2.4, label=label)
    ax.set_xticks(h)
    ax.set_xlabel('Giờ dự báo h (giờ)')
    ax.set_ylabel('RMSE (MW)')
    ax.set_ylim(0, None)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    out = FE / 'figures/horizon_rmse.png'
    fig.savefig(out, dpi=200)
    print('saved', out)


if __name__ == '__main__':
    main()
