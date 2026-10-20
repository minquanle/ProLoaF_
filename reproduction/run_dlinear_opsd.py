"""Apply the unchanged official DLinear architecture to OPSD, not reproduce its paper benchmark.

Adapter protocol: DE load only, train-only MinMax, 147 -> 24, chronological
60/20/20, Adam 0.001 fixed, batch 32, max 50 epochs, patience 7, three seeds.
All seeds are frozen on B before any C inference. No test-based tuning.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
PIN = '0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def windows(values):
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy())
    block = tensor.unfold(0, 171, 1)
    return TensorDataset(block[:, :147].unsqueeze(-1), block[:, 147:].unsqueeze(-1))


def metrics(actual, predicted, scale):
    error = (actual.astype(np.float64) - predicted.astype(np.float64)) * scale
    return {'RMSE_MW': float(np.sqrt(np.mean(error**2))),
            'MAE_MW': float(np.mean(np.abs(error))), 'Bias_actual_minus_predicted_MW': float(error.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = {'state': 'preparing', 'started_utc': datetime.now(timezone.utc).isoformat()}
    save_json(out/'status.json', status)
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        source = ROOT/'external/LTSF-Linear'
        head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
        assert head == PIN, 'Pinned upstream commit mismatch'
        changed = subprocess.check_output(['git','-C',str(source),'diff','HEAD','--','models/DLinear.py'], text=True)
        assert not changed, 'Official model has local changes'
        spec = importlib.util.spec_from_file_location('official_dlinear', source/'models/DLinear.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_config = SimpleNamespace(seq_len=147, pred_len=24, enc_in=1, individual=False)
        config = {'seeds':[1,2,3], 'history':147, 'horizon':24, 'batch_size':32,
                  'learning_rate':0.001, 'lr_schedule':'fixed', 'max_epochs':50, 'patience':7,
                  'loss':'MSE', 'optimizer':'Adam', 'target':'DE_load_actual_entsoe_transparency',
                  'input_features':['DE_load_actual_entsoe_transparency'], 'future_covariates':False,
                  'individual':False, 'decomposition_kernel':25, 'scaler_fit':'A only',
                  'selection':'B MSE only', 'test_used_for_selection':False,
                  'batching':'train shuffle=True; all loaders drop_last=False; validation/test ordered',
                  'source_repository':'https://github.com/cure-lab/LTSF-Linear', 'source_commit':head,
                  'source_model_sha256':sha(source/'models/DLinear.py'),
                  'paper_doi':'10.1609/aaai.v37i9.26317',
                  'scope':'Official architecture on OPSD; adapter training protocol, not original-paper benchmark.',
                  'comparison_limit':'ProLoaF has additional covariates and a different training loss. Not architecture-only comparison.',
                  'test_history':'C has been inspected in prior project experiments. No new hyperparameter choice is based on C here.',
                  'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,
                             'pandas':pd.__version__,'device':'cpu','threads':4}}
        datafile = ROOT/'paper_source/data/opsd.csv'
        frame = pd.read_csv(datafile, sep=';', index_col=0, parse_dates=True)
        assert len(frame)==43824 and frame.index.is_unique and frame.index.is_monotonic_increasing
        assert np.all(np.diff(frame.index.asi8)==3600*10**9)
        values = frame[config['target']].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all()
        a, b = int(len(frame)*0.6), int(len(frame)*0.8)
        minimum, scale = float(values[:a].min()), float(np.ptp(values[:a]))
        assert minimum == 31307.0 and scale == 46242.0
        scaled = ((values-minimum)/scale).astype(np.float32)
        train, val = windows(scaled[:a]), windows(scaled[a:b])
        assert len(train)==26124 and len(val)==8595
        config.update(data_sha256=sha(datafile), target_min_MW=minimum, target_range_MW=scale,
                      partitions={'A_rows':a, 'B_rows':b-a, 'C_rows':len(frame)-b,
                                  'A_windows':len(train),'B_windows':len(val),'C_windows':8595},
                      parameter_count=sum(p.numel() for p in module.Model(model_config).parameters()))
        save_json(out/'protocol.json', config)
        if args.dry_run:
            torch.manual_seed(1)
            test_model = module.Model(model_config)
            x, y = next(iter(DataLoader(train,batch_size=32)))
            prediction = test_model(x)
            assert prediction.shape == y.shape == (32,24,1)
            trend_input, trend = test_model.decompsition(x)
            torch.testing.assert_close(trend_input+trend,x)
            assert config['parameter_count']==7104
            status.update(state='dry_run_passed', output_shape=list(prediction.shape))
            save_json(out/'status.json',status)
            print('DRY RUN PASSED: source pinned, split correct, 7104 parameters, tensor and decomposition checks passed.',flush=True)
            return
        history, training_results = [], []
        for seed in config['seeds']:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = module.Model(model_config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            trainloader = DataLoader(train,batch_size=32,shuffle=True,drop_last=False)
            valloader = DataLoader(val,batch_size=512,shuffle=False,drop_last=False)
            best, stale, best_epoch = float('inf'), 0, 0
            checkpoint = out/f'dlinear_seed{seed}.pt'
            train_started = time.perf_counter()
            for epoch in range(1,config['max_epochs']+1):
                model.train(); train_total=0.0; count=0
                for x,y in trainloader:
                    optimizer.zero_grad(set_to_none=True)
                    prediction=model(x)
                    loss=torch.nn.functional.mse_loss(prediction,y)
                    if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                    loss.backward(); optimizer.step()
                    train_total += loss.item()*len(x); count+=len(x)
                model.eval(); val_sum=0.0; val_count=0
                with torch.no_grad():
                    for x,y in valloader:
                        val_sum += torch.nn.functional.mse_loss(model(x),y,reduction='sum').item()
                        val_count += y.numel()
                val_loss=val_sum/val_count
                if val_loss < best:
                    best,stale,best_epoch=val_loss,0,epoch
                    torch.save({'state_dict':copy.deepcopy(model.state_dict()),'seed':seed,
                                'best_epoch':epoch,'best_validation_mse':best,'protocol':config},checkpoint)
                else: stale+=1
                history.append({'seed':seed,'epoch':epoch,'train_mse':train_total/count,
                                'validation_B_mse':val_loss,'elapsed_seconds':time.perf_counter()-train_started})
                pd.DataFrame(history).to_csv(out/'epoch_history.csv',index=False)
                status.update(state='training',seed=seed,epoch=epoch,best_epoch=best_epoch,best_validation_mse=best)
                save_json(out/'status.json',status)
                print(f'seed={seed} epoch={epoch}/50 train={train_total/count:.7f} B={val_loss:.7f} best={best_epoch}',flush=True)
                if stale>=config['patience']: break
            training_results.append({'seed':seed,'best_epoch':best_epoch,'epochs':epoch,
                                     'best_validation_mse':best,'train_wall_seconds':time.perf_counter()-train_started,
                                     'checkpoint_sha256':sha(checkpoint)})
        # No C loader or prediction was constructed before all selections above were frozen.
        status.update(state='evaluating_C'); save_json(out/'status.json',status)
        test=windows(scaled[b:]); assert len(test)==8595
        loader=DataLoader(test,batch_size=512,shuffle=False,drop_last=False)
        actual=test.tensors[1][:,:,0].numpy()
        lstm_cache=ROOT/'artifacts/paper_reproduction/strict_602020_20261013/test_predictions.npz'
        lstm=np.load(lstm_cache)
        np.testing.assert_allclose(actual,lstm['actual_scaled'],atol=2e-7,rtol=0)
        indices=np.arange(8595)[:,None]+147+np.arange(24)[None,:]
        timestamps=frame.index[b:].to_numpy()[indices]
        np.testing.assert_allclose(actual*scale+minimum,values[b:][indices],atol=0.02,rtol=0)
        rows=[]; horizon_rows=[]
        for info in training_results:
            seed=info['seed']; model=module.Model(model_config)
            saved=torch.load(out/f'dlinear_seed{seed}.pt',map_location='cpu')
            model.load_state_dict(saved['state_dict']); model.eval()
            inf_start=time.perf_counter()
            with torch.no_grad(): predicted=torch.cat([model(x)[:,:,0] for x,_ in loader]).numpy()
            inf_time=time.perf_counter()-inf_start
            assert predicted.shape==(8595,24) and np.isfinite(predicted).all()
            np.savez_compressed(out/f'test_predictions_seed{seed}.npz',actual_scaled=actual,
                                mean_scaled=predicted,target_timestamps=timestamps,
                                checkpoint_sha256=info['checkpoint_sha256'])
            rows.append({'model':'DLinear_univariate','seed':seed,**info,**metrics(actual,predicted,scale),
                         'parameter_count':7104,'inference_C_seconds':inf_time})
            for h in range(24): horizon_rows.append({'model':'DLinear','seed':seed,'horizon':h+1,
                                                    **metrics(actual[:,h],predicted[:,h],scale)})
        summary=pd.DataFrame(rows); summary.to_csv(out/'test_summary_per_seed.csv',index=False)
        pd.DataFrame(horizon_rows).to_csv(out/'test_metrics_by_horizon.csv',index=False)
        comparisons=[{'model':'ProLoaF_LSTM_strict_multivariate','seed':1,
                      **metrics(actual,lstm['mean_scaled'],scale),'input_note':'5 historical features + future calendar; Gaussian NLL'},
                     {'model':'SeasonalNaive24','seed':None,
                      **metrics(actual,test.tensors[0][:,-24:,0].numpy(),scale),'input_note':'DE load only'},
                     {'model':'SimpleNaive','seed':None,
                      **metrics(actual,np.repeat(test.tensors[0][:,-1,0].numpy()[:,None],24,axis=1),scale),'input_note':'DE load only'}]
        comparisons.extend({'model':'DLinear_univariate','seed':r['seed'],
                            **{k:r[k] for k in ['RMSE_MW','MAE_MW','Bias_actual_minus_predicted_MW']},
                            'input_note':'DE load only; MSE'} for r in rows)
        pd.DataFrame(comparisons).to_csv(out/'comparison_point_metrics.csv',index=False)
        aggregate={k:{'mean':float(summary[k].mean()),'sample_std':float(summary[k].std(ddof=1))}
                   for k in ['RMSE_MW','MAE_MW','train_wall_seconds']}
        save_json(out/'aggregate_three_seeds.json',aggregate)
        status.update(state='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
                      total_wall_seconds=time.perf_counter()-started,results=rows,aggregate=aggregate)
        save_json(out/'status.json',status)
        print(pd.DataFrame(comparisons).to_string(index=False),flush=True)
        print('Three-seed mean/std: '+json.dumps(aggregate),flush=True)
    except BaseException:
        status.update(state='failed',error=traceback.format_exc());save_json(out/'status.json',status)
        raise


if __name__=='__main__': main()
