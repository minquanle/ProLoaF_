"""Read-only checks of persisted DLinear predictions, source, checkpoints and alignment."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from run_dlinear_opsd import metrics, windows

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    status = json.loads((out/'status.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    assert status['state'] == 'complete'
    model_file = ROOT/'external/LTSF-Linear/models/DLinear.py'
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == protocol['source_model_sha256']
    spec = importlib.util.spec_from_file_location('checked_dlinear', model_file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.set_num_threads(4)
    datafile = ROOT/'paper_source/data/opsd.csv'
    assert hashlib.sha256(datafile.read_bytes()).hexdigest() == protocol['data_sha256']
    frame = pd.read_csv(datafile, sep=';', index_col=0, parse_dates=True)
    target = frame[protocol['target']].to_numpy()
    minimum, scale = protocol['target_min_MW'], protocol['target_range_MW']
    test = windows(((target[35059:]-minimum)/scale).astype(np.float32))
    summary = pd.read_csv(out/'test_summary_per_seed.csv')
    lstm_root = ROOT/'artifacts/paper_reproduction/strict_602020_20261013'
    lstm = np.load(lstm_root/'test_predictions.npz')
    assert lstm['checkpoint_sha256'].item() == hashlib.sha256((lstm_root/'opsd_recurrent.pkl').read_bytes()).hexdigest()
    for seed in [1,2,3]:
        saved = np.load(out/f'test_predictions_seed{seed}.npz')
        row = summary[summary['seed']==seed].iloc[0]
        checkpoint = out/f'dlinear_seed{seed}.pt'
        assert saved['checkpoint_sha256'].item() == hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
        np.testing.assert_allclose(saved['actual_scaled'],lstm['actual_scaled'],rtol=0,atol=2e-7)
        assert saved['actual_scaled'].shape == saved['mean_scaled'].shape == (8595,24)
        expected_indices = 35059 + np.arange(8595)[:,None]+147+np.arange(24)[None,:]
        np.testing.assert_array_equal(saved['target_timestamps'],frame.index.to_numpy()[expected_indices])
        recomputed = metrics(saved['actual_scaled'],saved['mean_scaled'],scale)
        for key,value in recomputed.items(): np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-8)
        model = module.Model(SimpleNamespace(seq_len=147,pred_len=24,enc_in=1,individual=False))
        state = torch.load(checkpoint,map_location='cpu')
        model.load_state_dict(state['state_dict']);model.eval()
        assert state['best_epoch']==row['best_epoch']
        with torch.no_grad(): rebuilt=model(test.tensors[0][:256])[:,:,0].numpy()
        np.testing.assert_allclose(rebuilt,saved['mean_scaled'][:256],rtol=1e-5,atol=1e-6)
        print(f'PASS seed {seed}: checkpoint reload, source/data hashes, timestamps, target alignment, recomputed metrics.')
    print('ALL CHECKS PASSED. No files changed and no training performed.')


if __name__=='__main__': main()
