"""Train the frozen ProLoaF LSTM with A for gradients, B for selection, C for test.

No upstream files or previous checkpoint are modified. This is a corrected-split
experiment, not a claim about the code that generated the published figure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1, help='torch seed; 1 reproduces the first strict run.')
    parser.add_argument('--dry-run', action='store_true', help='Validate split and tensors without fitting.')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    # Refuse reuse: a run cannot silently overwrite a checkpoint or test result.
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)  # Upstream eventlog and TensorBoard files stay inside this run.
    status = {'state': 'preparing', 'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()}
    write_json(out / 'status.json', status)
    start = time.perf_counter()
    try:
        import numpy as np
        import pandas as pd
        import torch
        sys.path.insert(0, str(ROOT / 'paper_source/src'))
        import proloaf.datahandler as dh
        import proloaf.tensorloader as tl
        import proloaf.modelhandler as mh
        import proloaf.metrics as pm
        from calibrate_lstm_intervals import calculate_metrics, metrics_by_horizon

        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        config_file = ROOT / 'artifacts/paper_reproduction/training_config_used.json'
        old_checkpoint = ROOT / 'artifacts/paper_reproduction/opsd_recurrent.pkl'
        source_file = ROOT / 'paper_source/data/opsd.csv'
        config = json.loads(config_file.read_text(encoding='utf-8'))
        original_config = dict(config)
        config.update(output_path='./', log_path='./logs/', evaluation_path='./evaluation/')
        assert config['train_split'] == 0.6 and config['validation_split'] == 0.8
        assert config['exploration'] is False and config['max_epochs'] == 50
        frame = pd.read_csv(source_file, sep=';', index_col=0, parse_dates=True)
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        assert not frame.isna().any().any()
        train, validation, test = dh.split(frame, [0.6, 0.8])
        assert (len(train), len(validation), len(test)) == (26294, 8765, 8765)
        assert train.index[-1] < validation.index[0] < test.index[0]
        target = config['target_id'][0]
        scale = float(train[target].max() - train[target].min())
        minimum = float(train[target].min())
        meta = {
            'protocol': 'strict chronological 60/20/20', 'seed': args.seed,
            'source_commit': '1055d1fd3ae00ffce0b67eed03ec84a7410f5da8',
            'data_sha256': fingerprint(source_file), 'old_checkpoint_sha256': fingerprint(old_checkpoint),
            'base_config_sha256': fingerprint(config_file),
            'gradient_partition': 'A [0,60%)', 'selection_partition': 'B [60%,80%)',
            'final_test_partition': 'C [80%,100%)', 'scaler_fit_partition': 'A only',
            'test_used_for_selection': False, 'calibration_applied': False,
            'target_min_MW': minimum, 'target_range_MW': scale,
            'runtime': {'python': sys.version, 'torch': torch.__version__, 'device': 'cpu', 'threads': args.threads},
            'runtime_note': 'Four CPU threads for resource control; numerical results can differ from an older run with unspecified thread count.',
            'checkpoint_rule': 'Best B loss, including when the 50-epoch cap is reached.',
            'partitions': {name: {'rows': len(df), 'start': str(df.index[0]), 'end': str(df.index[-1]),
                                  'windows': len(df) - config['history_horizon'] - config['forecast_horizon'] + 1}
                           for name, df in [('A', train), ('B', validation), ('C', test)]},
        }
        write_json(out / 'protocol.json', meta)
        write_json(out / 'training_config_used.json', config)
        write_json(out / 'original_training_config.json', original_config)
        scaler = dh.MultiScaler(config['feature_groups'])

        def dataset(df, fit=False):
            return tl.TimeSeriesData(df.copy(), device='cpu', preparation_steps=[
                dh.set_to_hours,
                partial(dh.fill_if_missing, periodicity=config.get('periodicity', 24)),
                dh.add_cyclical_features, dh.add_onehot_features,
                scaler.fit_transform if fit else scaler.transform, dh.check_continuity,
            ], **config)

        # Tensorize A first, so B can only transform using the already fitted scaler.
        train_data = dataset(train, fit=True).to_tensor()
        validation_data = dataset(validation).to_tensor()
        assert len(train_data) == 26124 and len(validation_data) == 8595
        print('SPLIT CHECK PASSED: A=26294, B=8765, C=8765 hours; only B enters early stopping.', flush=True)
        if args.dry_run:
            enc, dec, y = next(iter(train_data.make_data_loader(batch_size=32, shuffle=False)))
            assert tuple(enc.shape) == (32, 147, 5)
            assert tuple(dec.shape) == (32, 24, 46) and tuple(y.shape) == (32, 24, 1)
            status.update(state='dry_run_passed', tensor_shapes=[list(enc.shape), list(dec.shape), list(y.shape)])
            write_json(out / 'status.json', status)
            print('DRY RUN PASSED: tensors valid; no fitting and no C tensors constructed.', flush=True)
            return

        history = []
        original_validate = mh.TrainingRun.validate
        original_save = mh.EarlyStopping.save_checkpoint

        def tracked_validate(run):
            result = original_validate(run)
            history.append({'epoch': len(history) + 1, 'train_loss': float(run.training_loss),
                            'validation_B_loss': float(run.validation_loss), 'elapsed_seconds': time.perf_counter() - start})
            pd.DataFrame(history).to_csv(out / 'epoch_history.csv', index=False)
            status.update(state='training', epoch=len(history), max_epochs=config['max_epochs'],
                          validation_B_loss=float(run.validation_loss))
            write_json(out / 'status.json', status)
            return result

        def tracked_save(stopper, loss, model):
            result = original_save(stopper, loss, model)
            # A durable copy of exactly the upstream-selected weights, not a new selection rule.
            torch.save(model.state_dict(), out / 'best_weights.pt')
            status.update(best_epoch=len(history), best_validation_B_loss=float(loss))
            write_json(out / 'status.json', status)
            return result

        mh.TrainingRun.validate = tracked_validate
        mh.EarlyStopping.save_checkpoint = tracked_save
        handler = mh.ModelHandler(work_dir=str(out), config=config, tuning_config=None,
                                  scalers=scaler, loss='NllGauss', loss_kwargs={}, device='cpu')
        try:
            handler.fit(train_data, validation_data)
        finally:
            mh.TrainingRun.validate = original_validate
            mh.EarlyStopping.save_checkpoint = original_save
        # Explicitly restore best B checkpoint also at max_epochs (upstream only restores on early stop).
        handler.model_wrap.model.load_state_dict(torch.load(out / 'best_weights.pt', map_location='cpu'))
        handler.model_wrap.last_training.validation_loss = status['best_validation_B_loss']
        handler.model_wrap.model.eval()
        checkpoint = out / 'opsd_recurrent.pkl'
        handler.save_current_model(str(checkpoint))
        write_json(out / 'trained_config.json', handler.get_config())
        status.update(state='evaluating_C', actual_epochs=len(history), training_wall_seconds=time.perf_counter()-start,
                      checkpoint_sha256=fingerprint(checkpoint))
        write_json(out / 'status.json', status)
        print('TRAIN DONE: best epoch {}; now evaluating C only.'.format(status['best_epoch']), flush=True)
        # Test construction and inference occur only after checkpoint selection is frozen.
        model = mh.ModelHandler.load_model(str(checkpoint), locate='cpu')
        model.to('cpu')
        model.model.eval()
        test_data = dataset(test)
        loader = test_data.make_data_loader(batch_size=256, shuffle=False, drop_last=False)
        target_batches, prediction_batches = [], []
        with torch.no_grad():
            for i, (enc, dec, y) in enumerate(loader):
                target_batches.append(y.cpu())
                prediction_batches.append(model.predict(enc, dec).cpu())
                if i % 8 == 0:
                    print('TEST: {} windows'.format(sum(len(t) for t in target_batches)), flush=True)
        targets, predictions = torch.cat(target_batches), torch.cat(prediction_batches)
        assert tuple(targets.shape) == (8595, 24, 1)
        actual = targets[:, :, 0].numpy()
        mean = predictions[:, :, 0].numpy()
        sigma = torch.exp(0.5 * predictions[:, :, 1]).numpy()
        assert np.isfinite(predictions.numpy()).all() and (sigma > 0).all()
        indices = np.arange(8595)[:, None] + config['history_horizon'] + np.arange(24)[None, :]
        np.testing.assert_allclose(actual * scale + minimum, test[target].to_numpy()[indices], atol=0.02, rtol=0)
        np.savez_compressed(out / 'test_predictions.npz', actual_scaled=actual, mean_scaled=mean,
                            sigma_scaled=sigma, checkpoint_sha256=fingerprint(checkpoint))
        quantiles = model.loss_metric.get_quantile_prediction(predictions=predictions, target=targets)
        paper_values = {metric.id: float(metric.from_quantiles(targets, quantiles, avg_over='all').item())
                        for metric in [pm.NllGauss(), pm.Mse(), pm.Rmse(), pm.Sharpness(), pm.Picp(),
                                       pm.Rae(), pm.Mae(), pm.Mis(), pm.Mase(), pm.PinnballLoss(), pm.Residuals()]}
        pd.DataFrame([paper_values], index=['opsd_recurrent']).to_csv(out / 'paper_metrics.csv', sep=';')
        new_metrics = calculate_metrics(actual, mean, sigma, 1.0)
        for key in ['RMSE', 'MAE', 'Sharpness', 'MIS_standard_95']:
            new_metrics[key + '_MW'] = new_metrics[key] * scale
        new_metrics.update(NllGauss=paper_values['NllGauss'], Residuals_MW=float((actual-mean).mean()*scale),
                           ProLoaF_legacy_MIS=paper_values['Mis'])
        assert np.isclose(new_metrics['PICP_percent'], paper_values['Picp'], rtol=1e-5)
        assert np.isclose(new_metrics['RMSE'], paper_values['Rmse'], rtol=1e-5)
        pd.DataFrame([new_metrics]).to_csv(out / 'test_summary.csv', index=False)
        horizon = metrics_by_horizon(actual, mean, sigma, 1.0, 'strict_602020_original_95')
        horizon.to_csv(out / 'test_metrics_by_horizon.csv', index=False)
        old = pd.read_csv(ROOT / 'artifacts/paper_reproduction/interval_calibration/test_calibrated_summary.csv').iloc[0].to_dict()
        for key in ['RMSE', 'MAE', 'Sharpness', 'MIS_standard_95']:
            old[key + '_MW'] = old[key] * scale
        comparison = pd.DataFrame([{'run': 'old_A_train_BplusC_selection', **old},
                                   {'run': 'new_A_train_B_selection_C_test', **new_metrics}])
        comparison.to_csv(out / 'comparison_old_vs_strict.csv', index=False)
        assert fingerprint(old_checkpoint) == meta['old_checkpoint_sha256']
        status.update(state='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                      total_wall_seconds=time.perf_counter()-start, test_summary=new_metrics)
        write_json(out / 'status.json', status)
        print(comparison[['run','RMSE_MW','MAE_MW','PICP_percent','Sharpness_MW','MIS_standard_95_MW']].to_string(index=False), flush=True)
    except BaseException:
        status.update(state='failed', error=traceback.format_exc())
        write_json(out / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
