"""Vietnamese figures for the final report and slides, from saved results only (no model run).

Output: artifacts/final_exam/figures/
  case_days.png      forecasts before/after improvement on three hard days of 2019
  lag_attribution.png Integrated Gradients by lag for DE load, before vs after
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / 'artifacts/final_exam'
OUT = FE / 'figures'
SCALE, MINIMUM, Z = 46242.0, 31307.0, 1.959963984540054
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def main():
    OUT.mkdir(exist_ok=True)
    tests = json.loads((FE / 'results/tests.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True, usecols=[0])
    test_index = frame.index[int(len(frame) * 0.8):]
    berlin = test_index[np.arange(8595) + 147].tz_localize('UTC').tz_convert('Europe/Berlin')
    runs = {'LSTM-chuẩn (giữa kỳ)': ('baseline', '#E48E7E'), 'LSTM-cuối kỳ': ('at_hol', '#3183A7')}
    days = [('Thứ Tư 01/05/2019 – Quốc tế Lao động', 5, 1), ('Thứ Năm 20/06/2019 – lễ Mình Thánh Chúa (một số bang)', 6, 20),
            ('Thứ Ba 24/12/2019 – đêm Giáng sinh', 12, 24)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (title, month, day) in zip(axes, days):
        i = int(np.where((berlin.month == month) & (berlin.day == day) & (berlin.hour == 0))[0][0])
        hours = np.arange(24)
        actual = None
        for label, (variant, color) in runs.items():
            d = np.load(FE / f'results/{variant}_ensemble_test.npz')
            c = tests[variant]['c_global_from_B']
            mu = d['mean_scaled'][i] * SCALE + MINIMUM
            half = Z * c * d['sigma_scaled'][i] * SCALE
            actual = d['actual_scaled'][i] * SCALE + MINIMUM
            rmse = np.sqrt(np.mean((actual - mu) ** 2))
            ax.fill_between(hours, (mu - half) / 1000, (mu + half) / 1000, color=color, alpha=0.18)
            ax.plot(hours, mu / 1000, color=color, lw=2, label=f'{label}: RMSE {rmse:,.0f} MW'.replace(',', '.'))
        ax.plot(hours, actual / 1000, color='k', lw=2, label='Thực tế')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Giờ trong ngày (giờ Đức)')
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.legend(fontsize=8, loc='upper left')
    axes[0].set_ylabel('Phụ tải (GW)')
    fig.tight_layout()
    fig.savefig(OUT / 'case_days.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for path, label, color in [(FE / 'xai/integrated_gradients_by_lag.csv', 'LSTM-chuẩn (giữa kỳ)', '#E48E7E'),
                               (FE / 'xai/at_hol/integrated_gradients_by_lag.csv', 'LSTM-cuối kỳ', '#3183A7')]:
        lag = pd.read_csv(path, index_col=0)
        ax.plot(lag.index, lag['DE load'], color=color, lw=2, label=label)
    ax.axvline(-24, color='gray', ls=':', lw=1)
    ax.text(-24, ax.get_ylim()[1] * 0.9, ' 24 giờ trước', color='gray', fontsize=8)
    ax.set_xlabel('Số giờ trước thời điểm dự báo')
    ax.set_ylabel('|IG| trung bình của phụ tải Đức')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'lag_attribution.png', dpi=170)
    plt.close(fig)
    print('saved', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
