"""XAI for the strict LSTM on the test year C. No training; saved checkpoints only.

1. Permutation importance (global, model-agnostic): shuffle one input group across
   test windows and measure the RMSE increase of the 5-seed ensemble.
2. Integrated Gradients (local -> global): attribution of every input cell
   (feature x hour) to the predicted mean mu and to the predicted log-variance v.
   Baseline = average test window. Completeness is checked (sum of attributions
   ~= f(x) - f(baseline)).
3. Occlusion saliency (perturbation, in the spirit of ProLoaF's interpreter.py):
   replace one encoder feature in one 24 h block by its baseline and measure |delta mu|.
4. Deletion test (faithfulness): occlude the top-k encoder cells ranked by |IG|
   versus k random cells; a faithful explanation must hurt RMSE more.
5. Case studies: best day, worst day and Christmas Eve 2019 with IG heatmaps.
Outputs: artifacts/final_exam/xai/
"""
import json
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'paper_source/src'))
import proloaf.datahandler as dh  # noqa: E402
import proloaf.modelhandler as mh  # noqa: E402
import proloaf.tensorloader as tl  # noqa: E402

ART = ROOT / 'artifacts/paper_reproduction'
RUNS = ROOT / 'artifacts/final_exam/runs'
# Usage: xai_analysis.py [baseline|at|at_hol]; baseline = midterm strict LSTM (default).
VARIANT = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
OUT = ROOT / 'artifacts/final_exam/xai' / ('' if VARIANT == 'baseline' else VARIANT)
FOLDERS = {1: 'strict_602020_20261013', 2: 'strict_602020_seed2', 3: 'strict_602020_seed3',
           4: 'strict_602020_seed4', 5: 'strict_602020_seed5'}
CHECKPOINTS = ({s: ART / f / 'opsd_recurrent.pkl' for s, f in FOLDERS.items()} if VARIANT == 'baseline'
               else {s: RUNS / f'{VARIANT}_seed{s}' / 'model.pkl' for s in FOLDERS})
CONFIG_FILE = (ART / 'training_config_used.json' if VARIANT == 'baseline'
               else RUNS / f'{VARIANT}_seed1' / 'training_config_used.json')
EXPLAIN_SEED = 2  # baseline: its RMSE (3260 MW) is closest to the ensemble (3375 MW); same seed for variants
SCALE, MINIMUM = 46242.0, 31307.0
SHORT = {'AT_load_actual_entsoe_transparency': 'AT load', 'DE_load_actual_entsoe_transparency': 'DE load',
         'DE_temperature': 'DE temperature', 'DE_radiation_direct_horizontal': 'DE radiation direct',
         'DE_radiation_diffuse_horizontal': 'DE radiation diffuse'}
rng = np.random.default_rng(0)


def load_data():
    config = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
    frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True)
    if 'is_holiday' in config['decoder_features']:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_lstm_variant import add_calendar_flags
        frame = add_calendar_flags(frame)
    train, _, test = dh.split(frame, [0.6, 0.8])
    scaler = dh.MultiScaler(config['feature_groups'])

    def dataset(df, fit=False):
        return tl.TimeSeriesData(df.copy(), device='cpu', preparation_steps=[
            dh.set_to_hours, partial(dh.fill_if_missing, periodicity=24),
            dh.add_cyclical_features, dh.add_onehot_features,
            scaler.fit_transform if fit else scaler.transform, dh.check_continuity], **config)

    dataset(train, fit=True).to_tensor()
    data = dataset(test).to_tensor()
    loader = data.make_data_loader(batch_size=10000, shuffle=False, drop_last=False)
    enc, dec, y = next(iter(loader))
    assert enc.shape[0] == 8595
    times = test.index[np.arange(8595) + 147]  # first target hour (UTC) of each window
    return config, enc, dec, y[:, :, 0], times


def load_model(seed):
    model = mh.ModelHandler.load_model(str(CHECKPOINTS[seed]), locate='cpu')
    model.to('cpu')
    net = model.model_wrap.model if hasattr(model, 'model_wrap') else model.model
    net.eval()
    return net


def forward(net, enc, dec, batch=2048):
    outs = []
    with torch.no_grad():
        for i in range(0, len(enc), batch):
            outs.append(net(enc[i:i + batch], dec[i:i + batch])[0])
    return torch.cat(outs)


def rmse(y, mu):
    return float(torch.sqrt(torch.mean((y - mu) ** 2)) * SCALE)


def permutation_importance(nets, enc, dec, y, enc_names, dec_names, repeats=3):
    groups = {SHORT[n]: ('enc', [i]) for i, n in enumerate(enc_names)}
    for label, prefix in [('Calendar: hour one-hot', 'hour_'), ('Calendar: weekday one-hot', 'weekday_'),
                          ('Calendar: month one-hot', 'month_')]:
        groups[label] = ('dec', [i for i, n in enumerate(dec_names) if n.startswith(prefix) and not n.endswith('sin')])
    groups['Calendar: sin features'] = ('dec', [i for i, n in enumerate(dec_names) if n.endswith('sin')])
    if 'is_holiday' in dec_names:
        groups['Calendar: holiday flags'] = ('dec', [dec_names.index('is_holiday'), dec_names.index('is_xmas')])
    groups['All calendar (decoder)'] = ('dec', list(range(len(dec_names))))

    def ensemble_rmse(e, d):
        return rmse(y, torch.stack([forward(n, e, d)[:, :, 0] for n in nets]).mean(0))

    base = ensemble_rmse(enc, dec)
    rows = []
    for label, (part, cols) in groups.items():
        scores = []
        for _ in range(repeats):
            perm = torch.as_tensor(rng.permutation(len(enc)), dtype=torch.long)
            e, d = enc.clone(), dec.clone()
            target = e if part == 'enc' else d
            cols_t = torch.as_tensor(cols, dtype=torch.long)
            target[:, :, cols_t] = target[perm][:, :, cols_t]
            scores.append(ensemble_rmse(e, d) - base)
        rows.append({'group': label, 'RMSE_increase_MW': float(np.mean(scores)), 'std_MW': float(np.std(scores))})
        print('permutation', label, round(rows[-1]['RMSE_increase_MW'], 1), flush=True)
    return base, pd.DataFrame(rows).sort_values('RMSE_increase_MW', ascending=False)


def integrated_gradients(net, enc, dec, enc_base, dec_base, output, steps=32):
    """IG of sum over the 24 horizons of mu (output=0) or log-variance (output=1)."""
    total_e, total_d = torch.zeros_like(enc), torch.zeros_like(dec)
    for alpha in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:
        e = (enc_base + alpha * (enc - enc_base)).requires_grad_(True)
        d = (dec_base + alpha * (dec - dec_base)).requires_grad_(True)
        out = net(e, d)[0][:, :, output].sum()
        ge, gd = torch.autograd.grad(out, [e, d])
        total_e += ge
        total_d += gd
    attr_e = (enc - enc_base) * total_e / steps
    attr_d = (dec - dec_base) * total_d / steps
    with torch.no_grad():
        gap = net(enc, dec)[0][:, :, output].sum(1) - net(enc_base, dec_base)[0][:, :, output].sum(1)
    return attr_e, attr_d, gap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config, enc, dec, y, times = load_data()
    enc_names, dec_names = config['encoder_features'], config['decoder_features']
    nets = {s: load_model(s) for s in FOLDERS}
    net = nets[EXPLAIN_SEED]
    torch.backends.cudnn.enabled = False

    # 1. Permutation importance on the ensemble
    base_rmse, perm = permutation_importance(list(nets.values()), enc, dec, y, enc_names, dec_names)
    perm.to_csv(OUT / 'permutation_importance.csv', index=False)
    plt.figure(figsize=(8, 4.2))
    plt.barh(perm['group'][::-1], perm['RMSE_increase_MW'][::-1], xerr=perm['std_MW'][::-1], color='#3183A7')
    plt.xlabel(f'RMSE increase when shuffled (MW); ensemble RMSE = {base_rmse:.0f} MW')
    plt.title('Permutation importance, LSTM ensemble, test year 2019')
    plt.tight_layout()
    plt.savefig(OUT / 'permutation_importance.png', dpi=150)
    plt.close()

    # 2. Integrated Gradients on a sample of windows (one per day + random extra)
    sample = np.unique(np.concatenate([np.arange(0, 8595, 24), rng.choice(8595, 200, replace=False)]))
    enc_base = enc.mean(0, keepdim=True).expand(len(sample), -1, -1)
    dec_base = dec.mean(0, keepdim=True).expand(len(sample), -1, -1)
    results = {}
    for output, name in [(0, 'mu'), (1, 'logvar')]:
        attr_e, attr_d, gap = integrated_gradients(net, enc[sample], dec[sample], enc_base, dec_base, output)
        completeness = float(torch.mean(torch.abs(attr_e.sum((1, 2)) + attr_d.sum((1, 2)) - gap)) /
                             torch.mean(torch.abs(gap)))
        results[name] = (attr_e.abs().mean(0).numpy(), attr_d.abs().mean(0).numpy(), completeness)
        print('IG', name, 'relative completeness error', round(completeness, 4), flush=True)
    rows = []
    for name, (ae, ad, comp) in results.items():
        for i, n in enumerate(enc_names):
            rows.append({'output': name, 'part': 'encoder', 'feature': SHORT[n], 'mean_abs_attribution': float(ae[:, i].sum())})
        for label, prefix in [('hour one-hot', 'hour_'), ('weekday one-hot', 'weekday_'), ('month one-hot', 'month_')]:
            cols = [i for i, n in enumerate(dec_names) if n.startswith(prefix) and not n.endswith('sin')]
            rows.append({'output': name, 'part': 'decoder', 'feature': label, 'mean_abs_attribution': float(ad[:, cols].sum())})
        cols = [i for i, n in enumerate(dec_names) if n.endswith('sin')]
        rows.append({'output': name, 'part': 'decoder', 'feature': 'sin features', 'mean_abs_attribution': float(ad[:, cols].sum())})
        if 'is_holiday' in dec_names:
            cols = [dec_names.index('is_holiday'), dec_names.index('is_xmas')]
            rows.append({'output': name, 'part': 'decoder', 'feature': 'holiday flags',
                         'mean_abs_attribution': float(ad[:, cols].sum())})
    ig = pd.DataFrame(rows)
    ig['share_percent'] = ig.groupby('output')['mean_abs_attribution'].transform(lambda v: 100 * v / v.sum())
    ig.to_csv(OUT / 'integrated_gradients_feature_share.csv', index=False)
    json.dump({k: v[2] for k, v in results.items()}, open(OUT / 'ig_completeness.json', 'w'), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, name, title in [(axes[0], 'mu', 'Prediction mean mu'), (axes[1], 'logvar', 'Uncertainty (log-variance)')]:
        part = ig[ig['output'] == name].sort_values('share_percent')
        ax.barh(part['feature'], part['share_percent'], color=['#E48E7E' if p == 'decoder' else '#3183A7' for p in part['part']])
        ax.set_xlabel('share of |IG| attribution (%)')
        ax.set_title(title)
    fig.suptitle(f'Integrated Gradients, LSTM seed {EXPLAIN_SEED}, {len(sample)} test windows (blue = past, red = calendar)')
    fig.tight_layout()
    fig.savefig(OUT / 'integrated_gradients_features.png', dpi=150)
    plt.close(fig)

    # attribution of encoder inputs by lag (hours before the forecast origin)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ae = results['mu'][0]
    lags = np.arange(147, 0, -1)
    for i, n in enumerate(enc_names):
        ax.plot(-lags, ae[:, i], label=SHORT[n])
    ax.set_xlabel('hours before forecast origin')
    ax.set_ylabel('mean |IG| on mu')
    ax.set_title('Which past hours matter: peaks at lag 24 h and 168 h would show daily/weekly memory')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'integrated_gradients_by_lag.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(ae, columns=[SHORT[n] for n in enc_names], index=-lags).to_csv(OUT / 'integrated_gradients_by_lag.csv')

    # 3. Occlusion saliency: encoder feature x 24 h block, on the same sample
    mu0 = forward(net, enc[sample], dec[sample])[:, :, 0]
    mean_enc = enc.mean(0)
    blocks = [(max(0, 147 - 24 * (b + 1)), 147 - 24 * b) for b in range(7)]
    occ = np.zeros((len(enc_names), len(blocks)))
    for i in range(len(enc_names)):
        for b, (lo, hi) in enumerate(blocks):
            e = enc[sample].clone()
            e[:, lo:hi, i] = mean_enc[lo:hi, i]
            occ[i, b] = float(torch.mean(torch.abs(forward(net, e, dec[sample])[:, :, 0] - mu0)) * SCALE)
    occ_df = pd.DataFrame(occ, index=[SHORT[n] for n in enc_names],
                          columns=[f'{lo - 147}..{hi - 147} h' for lo, hi in blocks])
    occ_df.to_csv(OUT / 'occlusion_saliency.csv')
    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(occ, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(blocks)))
    ax.set_xticklabels(occ_df.columns, fontsize=8)
    ax.set_yticks(range(len(enc_names)))
    ax.set_yticklabels(occ_df.index)
    for i in range(occ.shape[0]):
        for b in range(occ.shape[1]):
            ax.text(b, i, f'{occ[i, b]:.0f}', ha='center', va='center', color='w', fontsize=8)
    fig.colorbar(im, label='mean |change of mu| (MW)')
    ax.set_title('Occlusion saliency: replace one input block by its average')
    fig.tight_layout()
    fig.savefig(OUT / 'occlusion_saliency.png', dpi=150)
    plt.close(fig)

    # 4. Deletion test on all test windows
    rank = np.argsort(results['mu'][0].ravel())[::-1]
    rows = []
    full_mu = forward(net, enc, dec)[:, :, 0]
    for k in [10, 25, 50, 100, 200]:
        for kind in ['top-|IG|', 'random']:
            cells = rank[:k] if kind == 'top-|IG|' else rng.choice(rank.size, k, replace=False)
            e = enc.clone()
            t_idx, f_idx = (torch.as_tensor(a, dtype=torch.long) for a in np.unravel_index(cells, (147, len(enc_names))))
            e[:, t_idx, f_idx] = mean_enc[t_idx, f_idx]
            rows.append({'k_cells': k, 'cells_removed': kind,
                         'RMSE_MW': rmse(y, forward(net, e, dec)[:, :, 0]), 'RMSE_full_MW': rmse(y, full_mu)})
    deletion = pd.DataFrame(rows)
    deletion.to_csv(OUT / 'deletion_test.csv', index=False)
    print(deletion.round(1).to_string(index=False), flush=True)

    # 5. Case studies
    ens = torch.stack([forward(n, enc, dec) for n in nets.values()])
    mu_e = ens[:, :, :, 0].mean(0)
    sd_e = torch.sqrt((torch.exp(ens[:, :, :, 1]) + ens[:, :, :, 0] ** 2).mean(0) - mu_e ** 2)
    daily = np.arange(0, 8595, 24)
    day_err = torch.sqrt(torch.mean((y[daily] - mu_e[daily]) ** 2, 1)).numpy() * SCALE
    berlin = times.tz_localize('UTC').tz_convert('Europe/Berlin')
    xmas = int(np.where((berlin.month == 12) & (berlin.day == 24) & (berlin.hour == 0))[0][0])
    cases = {'best day': int(daily[np.argmin(day_err)]), 'worst day': int(daily[np.argmax(day_err)]),
             'Christmas Eve 2019': xmas}
    case_rows = []
    fig, axes = plt.subplots(len(cases), 2, figsize=(13, 3.4 * len(cases)), gridspec_kw={'width_ratios': [1, 1.4]})
    for row, (label, i) in enumerate(cases.items()):
        idx = [i]
        ae, ad, _ = integrated_gradients(net, enc[idx], dec[idx], enc.mean(0, keepdim=True), dec.mean(0, keepdim=True), 0)
        hours = pd.date_range(berlin[i], periods=24, freq='h')
        actual = y[i].numpy() * SCALE + MINIMUM
        mu = mu_e[i].numpy() * SCALE + MINIMUM
        sd = sd_e[i].numpy() * SCALE
        err = float(np.sqrt(np.mean((actual - mu) ** 2)))
        case_rows.append({'case': label, 'start_berlin': str(berlin[i]), 'RMSE_MW': err,
                          'top_input': SHORT[enc_names[int(np.argmax(ae[0].abs().sum(0)))]]})
        ax = axes[row, 0]
        ax.fill_between(hours, mu - 1.96 * sd, mu + 1.96 * sd, color='#3183A7', alpha=0.25, label='95% interval')
        ax.plot(hours, mu, color='#3183A7', label='forecast mu')
        ax.plot(hours, actual, color='k', label='actual')
        ax.set_title(f'{label}: {berlin[i]:%a %d %b %Y}, RMSE {err:.0f} MW', fontsize=10)
        ax.tick_params(axis='x', labelsize=7)
        ax.legend(fontsize=7)
        heat = ae[0, -72:, :].T.numpy() * SCALE
        ax = axes[row, 1]
        lim = np.abs(heat).max()
        im = ax.imshow(heat, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto', extent=[-72, 0, len(enc_names) - 0.5, -0.5])
        ax.set_yticks(range(len(enc_names)))
        ax.set_yticklabels([SHORT[n] for n in enc_names], fontsize=8)
        ax.set_xlabel('hours before forecast origin (last 72 h)')
        ax.set_title('IG on sum of mu (MW): red pushes forecast up, blue down', fontsize=10)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / 'case_studies.png', dpi=150)
    plt.close(fig)
    pd.DataFrame(case_rows).to_csv(OUT / 'case_studies.csv', index=False)
    print(pd.DataFrame(case_rows).to_string(index=False))
    print(perm.round(1).to_string(index=False))
    print(ig.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
