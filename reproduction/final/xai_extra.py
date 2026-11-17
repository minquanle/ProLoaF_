"""XAI checks added after the report review (no training).

1. Encoder gate statistics for every seed of a variant: how saturated the LSTM gates are and
   how much of the input pre-activation comes from each encoder feature. Tests the claim that
   the unscaled Austrian load saturates the encoder.
2. Integrated Gradients feature shares for all five seeds (not only seed 2).
3. Deletion test for all five seeds, with 20 random draws, and an extended version that also
   ranks and removes decoder (calendar) cells.

Usage: xai_extra.py <variant> [<variant> ...]   (baseline, at, at_hol, hol, noat)
Output: artifacts/final_exam/xai/extra_<variant>.json
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
SCALE = 46242.0


def gate_stats(net, enc, names):
    core = net.encoder.core
    w_ih, w_hh = core.weight_ih_l0, core.weight_hh_l0
    b = core.bias_ih_l0 + core.bias_hh_l0
    hidden = w_hh.shape[1]
    h = torch.zeros(len(enc), hidden)
    c = torch.zeros(len(enc), hidden)
    sat = {k: [] for k in 'ifo'}
    cell_sat, pre_abs, contrib = [], [], torch.zeros(len(names))
    with torch.no_grad():
        for t in range(enc.shape[1]):
            x = enc[:, t]
            pre = x @ w_ih.T + h @ w_hh.T + b
            i, f, g, o = pre.chunk(4, dim=1)
            i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            for k, v in zip('ifo', (i, f, o)):
                sat[k].append(float(((v < 0.05) | (v > 0.95)).float().mean()))
            cell_sat.append(float((g.abs() > 0.95).float().mean()))
            pre_abs.append(float(pre.abs().mean()))
            contrib += (x.abs().unsqueeze(1) * w_ih.abs().unsqueeze(0)).mean((0, 1))
    contrib = contrib / contrib.sum()
    return {'saturated_input_gate': float(np.mean(sat['i'])), 'saturated_forget_gate': float(np.mean(sat['f'])),
            'saturated_output_gate': float(np.mean(sat['o'])), 'saturated_cell_candidate': float(np.mean(cell_sat)),
            'mean_abs_preactivation': float(np.mean(pre_abs)),
            'input_preactivation_share': {n: float(v) for n, v in zip(names, contrib)}}


def run(variant):
    sys.argv = ['xai_analysis.py', variant]
    xa = importlib.reload(importlib.import_module('xai_analysis'))
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = False
    config, enc, dec, y, _ = xa.load_data()
    enc_names, dec_names = config['encoder_features'], config['decoder_features']
    short = [xa.SHORT[n] for n in enc_names]
    rng = np.random.default_rng(0)
    sample = np.unique(np.concatenate([np.arange(0, 8595, 24), rng.choice(8595, 200, replace=False)]))
    mean_enc, mean_dec = enc.mean(0), dec.mean(0)
    enc_base = mean_enc.unsqueeze(0).expand(len(sample), -1, -1)
    dec_base = mean_dec.unsqueeze(0).expand(len(sample), -1, -1)
    out = {'variant': variant, 'seeds': {}}
    for seed in xa.FOLDERS:
        net = xa.load_model(seed)
        res = {'gates': gate_stats(net, enc[sample], short)}
        ae, ad, _ = xa.integrated_gradients(net, enc[sample], dec[sample], enc_base, dec_base, 0)
        ae, ad = ae.abs().mean(0).numpy(), ad.abs().mean(0).numpy()
        total = ae.sum() + ad.sum()
        res['ig_share_percent'] = {**{n: float(ae[:, i].sum() / total * 100) for i, n in enumerate(short)},
                                   'calendar (decoder)': float(ad.sum() / total * 100)}
        full = xa.rmse(y, xa.forward(net, enc, dec)[:, :, 0])
        res['rmse_full_MW'] = full
        rank = np.argsort(ae.ravel())[::-1]
        deletion = {}
        for k in [10, 50, 200]:
            def masked(cells):
                e = enc.clone()
                t_idx, f_idx = (torch.as_tensor(a, dtype=torch.long) for a in np.unravel_index(cells, ae.shape))
                e[:, t_idx, f_idx] = mean_enc[t_idx, f_idx]
                return xa.rmse(y, xa.forward(net, e, dec)[:, :, 0])
            rand = [masked(rng.choice(rank.size, k, replace=False)) for _ in range(20)]
            deletion[str(k)] = {'top_IG_MW': masked(rank[:k]), 'random_mean_MW': float(np.mean(rand)),
                                'random_std_MW': float(np.std(rand)), 'random_max_MW': float(np.max(rand))}
        res['deletion_encoder'] = deletion
        if seed == xa.EXPLAIN_SEED:
            both = np.concatenate([ae.ravel(), ad.ravel()])
            rank_all = np.argsort(both)[::-1]
            n_enc = ae.size

            def masked_all(cells):
                e, d = enc.clone(), dec.clone()
                ce, cd = cells[cells < n_enc], cells[cells >= n_enc] - n_enc
                if len(ce):
                    t_idx, f_idx = (torch.as_tensor(a, dtype=torch.long) for a in np.unravel_index(ce, ae.shape))
                    e[:, t_idx, f_idx] = mean_enc[t_idx, f_idx]
                if len(cd):
                    t_idx, f_idx = (torch.as_tensor(a, dtype=torch.long) for a in np.unravel_index(cd, ad.shape))
                    d[:, t_idx, f_idx] = mean_dec[t_idx, f_idx]
                return xa.rmse(y, xa.forward(net, e, d)[:, :, 0])
            ext = {}
            for k in [10, 50, 200]:
                rand = [masked_all(rng.choice(both.size, k, replace=False)) for _ in range(20)]
                top = rank_all[:k]
                ext[str(k)] = {'top_IG_MW': masked_all(top), 'top_cells_in_decoder': int((top >= n_enc).sum()),
                               'random_mean_MW': float(np.mean(rand)), 'random_std_MW': float(np.std(rand))}
            res['deletion_all_cells'] = {'cells_total': int(both.size), **ext}
        out['seeds'][str(seed)] = res
        print(variant, 'seed', seed, json.dumps({k: res[k] for k in ['rmse_full_MW', 'ig_share_percent']}), flush=True)
        print('   gates', json.dumps(res['gates']), flush=True)
        print('   deletion', json.dumps(res['deletion_encoder']), flush=True)
    target = ROOT / 'artifacts/final_exam/xai' / f'extra_{variant}.json'
    target.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('saved', target)


if __name__ == '__main__':
    for v in sys.argv[1:]:
        run(v)
