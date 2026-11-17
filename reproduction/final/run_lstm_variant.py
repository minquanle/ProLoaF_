"""Final-exam LSTM variants on the strict 60/20/20 split (A train, B select, C test).

Variants change only the input data, never the architecture or training settings:
  at      : Austrian load joins the min-max scaled group (it was left unscaled).
  at_hol  : as `at`, plus two calendar flags known in advance for the decoder:
            is_holiday (German national holiday, Berlin local date) and
            is_xmas (24 Dec - 1 Jan, Berlin local date).
Control variants that complete the design (added after review):
  hol     : holiday flags only; Austrian load stays unscaled as in the original code
            (the missing cell of the 2x2 design C1 x C2).
  noat    : Austrian load removed from the encoder inputs (tests whether the unscaled
            Austrian load harms the model, and whether the series is needed at all).

Outputs per run: checkpoint, epoch history, B and C predictions (.npz), C summary.
Selection uses B only; C is evaluated once after the checkpoint is frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reproduction'))
AT = 'AT_load_actual_entsoe_transparency'


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def add_calendar_flags(frame):
    import holidays
    local = frame.index.tz_localize('UTC').tz_convert('Europe/Berlin')
    de = holidays.Germany(years=range(local.year.min(), local.year.max() + 1))
    frame = frame.copy()
    frame['is_holiday'] = [float(d in de) for d in local.date]
    frame['is_xmas'] = [float((m == 12 and d >= 24) or (m == 1 and d == 1)) for m, d in zip(local.month, local.day)]
    return frame


def variant_config(config, variant):
    config = deepcopy(config)
    groups = {g['name']: g for g in config['feature_groups']}
    if variant in ('at', 'at_hol'):
        groups['main']['features'] = [AT] + groups['main']['features']
    if variant == 'noat':
        config['encoder_features'] = [f for f in config['encoder_features'] if f != AT]
    if variant in ('at_hol', 'hol'):
        groups['aux']['features'] += ['is_holiday', 'is_xmas']
        config['decoder_features'] = config['decoder_features'] + ['is_holiday', 'is_xmas']
    return config


def predict(model, data):
    import torch
    loader = data.make_data_loader(batch_size=256, shuffle=False, drop_last=False)
    ys, ps = [], []
    with torch.no_grad():
        for enc, dec, y in loader:
            ys.append(y)
            ps.append(model.predict(enc, dec))
    y, p = torch.cat(ys), torch.cat(ps)
    return y[:, :, 0].numpy(), p[:, :, 0].numpy(), torch.exp(0.5 * p[:, :, 1]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['at', 'at_hol', 'hol', 'noat'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-epochs', type=int, default=None, help='only for smoke tests')
    args = parser.parse_args()
    out = ROOT / 'artifacts/final_exam/runs' / f'{args.variant}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    status = {'state': 'preparing', 'variant': args.variant, 'seed': args.seed, 'pid': os.getpid()}
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
        from calibrate_lstm_intervals import calculate_metrics

        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        base = json.loads((ROOT / 'artifacts/paper_reproduction/training_config_used.json').read_text(encoding='utf-8'))
        config = variant_config(base, args.variant)
        config.update(output_path='./', log_path='./logs/', evaluation_path='./evaluation/')
        if args.max_epochs:
            config['max_epochs'] = args.max_epochs
        frame = pd.read_csv(ROOT / 'paper_source/data/opsd.csv', sep=';', index_col=0, parse_dates=True)
        if args.variant in ('at_hol', 'hol'):
            frame = add_calendar_flags(frame)
        train, validation, test = dh.split(frame, [0.6, 0.8])
        assert (len(train), len(validation), len(test)) == (26294, 8765, 8765)
        target = config['target_id'][0]
        minimum = float(train[target].min())
        scale = float(train[target].max() - train[target].min())
        write_json(out / 'training_config_used.json', config)
        scaler = dh.MultiScaler(config['feature_groups'])

        def dataset(df, fit=False):
            return tl.TimeSeriesData(df.copy(), device='cpu', preparation_steps=[
                dh.set_to_hours, partial(dh.fill_if_missing, periodicity=24),
                dh.add_cyclical_features, dh.add_onehot_features,
                scaler.fit_transform if fit else scaler.transform, dh.check_continuity,
            ], **config)

        train_data = dataset(train, fit=True).to_tensor()
        validation_data = dataset(validation).to_tensor()

        history = []
        original_validate = mh.TrainingRun.validate
        original_save = mh.EarlyStopping.save_checkpoint

        def tracked_validate(run):
            result = original_validate(run)
            history.append({'epoch': len(history) + 1, 'train_loss': float(run.training_loss),
                            'validation_B_loss': float(run.validation_loss),
                            'elapsed_seconds': time.perf_counter() - start})
            pd.DataFrame(history).to_csv(out / 'epoch_history.csv', index=False)
            status.update(state='training', epoch=len(history))
            write_json(out / 'status.json', status)
            print(f"epoch {len(history)} train {run.training_loss:.4f} B {run.validation_loss:.4f}", flush=True)
            return result

        def tracked_save(stopper, loss, model):
            result = original_save(stopper, loss, model)
            torch.save(model.state_dict(), out / 'best_weights.pt')
            status.update(best_epoch=len(history), best_validation_B_loss=float(loss))
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
        handler.model_wrap.model.load_state_dict(torch.load(out / 'best_weights.pt', map_location='cpu'))
        handler.model_wrap.last_training.validation_loss = status['best_validation_B_loss']
        handler.model_wrap.model.eval()
        handler.save_current_model(str(out / 'model.pkl'))
        model = mh.ModelHandler.load_model(str(out / 'model.pkl'), locate='cpu')
        model.to('cpu')
        model.model.eval()

        yb, mb, sb = predict(model, validation_data)
        np.savez_compressed(out / 'validation_predictions.npz', actual_scaled=yb, mean_scaled=mb, sigma_scaled=sb)
        yc, mc, sc = predict(model, dataset(test).to_tensor())
        assert yc.shape == (8595, 24)
        idx = np.arange(8595)[:, None] + config['history_horizon'] + np.arange(24)[None, :]
        np.testing.assert_allclose(yc * scale + minimum, test[target].to_numpy()[idx], atol=0.02, rtol=0)
        np.savez_compressed(out / 'test_predictions.npz', actual_scaled=yc, mean_scaled=mc, sigma_scaled=sc)
        summary = calculate_metrics(yc, mc, sc, 1.0)
        for key in ['RMSE', 'MAE', 'Sharpness', 'MIS_standard_95']:
            summary[key + '_MW'] = summary[key] * scale
        pd.DataFrame([summary]).to_csv(out / 'test_summary.csv', index=False)
        status.update(state='complete', actual_epochs=len(history), wall_seconds=time.perf_counter() - start,
                      test_summary=summary)
        write_json(out / 'status.json', status)
        print('DONE', json.dumps({k: summary[k] for k in ['RMSE_MW', 'MAE_MW', 'PICP_percent']}), flush=True)
    except BaseException:
        status.update(state='failed', error=traceback.format_exc())
        write_json(out / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
