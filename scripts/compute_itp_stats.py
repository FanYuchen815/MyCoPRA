#!/usr/bin/env python3
"""Compute ITP weights for a dataset using a trained model checkpoint.

Example:
    python scripts/compute_itp_stats.py --ckpt outputs/ablation/full/model.pt --model_config config/models/prorna_ssdn.yml --data_config config/datasets/PRA310.yml --out outputs/itp_weights_full.csv --fold 0
"""
import os
import sys
# Ensure reasonable thread-related environment variables to avoid libgomp errors
try:
    if int(os.environ.get('OMP_NUM_THREADS', '0')) <= 0:
        os.environ['OMP_NUM_THREADS'] = '4'
except Exception:
    os.environ['OMP_NUM_THREADS'] = '4'
try:
    if int(os.environ.get('MKL_NUM_THREADS', '0')) <= 0:
        os.environ['MKL_NUM_THREADS'] = '4'
except Exception:
    os.environ['MKL_NUM_THREADS'] = '4'
os.environ.setdefault('NUMEXPR_MAX_THREADS', '8')

# Ensure repo root is on python path so `models` is importable when script run from repo root
sys.path.insert(0, os.getcwd())

import argparse
import yaml
from easydict import EasyDict
import importlib
import pkgutil
import torch
from pathlib import Path
import csv
import traceback
import torch

from models.register import ModelRegister
from pl_modules.data_module import DataModule


def load_model_from_cfg_and_ckpt(model_cfg_path, ckpt_path, device='cpu'):
    model_cfg = yaml.safe_load(open(model_cfg_path))
    model_args = model_cfg.get('model', {})
    # Ensure model classes are registered by importing common model modules
    try:
        importlib.import_module('models.model')
    except Exception:
        pass
    try:
        importlib.import_module('models.esm_rinalmo_seq')
    except Exception:
        pass
    try:
        importlib.import_module('models.benchmarks')
    except Exception:
        pass
    try:
        importlib.import_module('models.baselines')
    except Exception:
        pass

    register = ModelRegister()
    model_type = model_args.get('model_type')
    if model_type is None:
        raise ValueError('model_type not found in model config')
    if model_type not in register:
        # help the user by listing available registered models
        available = list(register.keys())
        raise KeyError(f"Model type '{model_type}' not registered. Available: {available}")
    model_cls = register[model_type]
    model = model_cls(**model_args)
    # load checkpoint
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model' in ckpt):
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            # try nested state
            try:
                model.load_state_dict(ckpt, strict=False)
            except Exception:
                pass
    else:
        try:
            model.load_state_dict(ckpt, strict=False)
        except Exception:
            # ckpt might be a pickled model object
            if hasattr(ckpt, 'state_dict'):
                model.load_state_dict(ckpt.state_dict(), strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model_config', default='config/models/prorna_ssdn.yml')
    p.add_argument('--data_config', default='config/datasets/PRA310.yml')
    p.add_argument('--out', default='outputs/itp_weights.csv')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    model = load_model_from_cfg_and_ckpt(args.model_config, args.ckpt, device=args.device)

    def _move_batch_to_device(batch, device):
        dev = torch.device(device)
        for k, v in list(batch.items()):
            try:
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(dev)
                elif isinstance(v, (list, tuple)):
                    # Move any tensor elements inside lists/tuples
                    moved = False
                    new_elems = []
                    for elem in v:
                        if isinstance(elem, torch.Tensor):
                            new_elems.append(elem.to(dev))
                            moved = True
                        else:
                            new_elems.append(elem)
                    if moved:
                        try:
                            batch[k] = type(v)(new_elems)
                        except Exception:
                            batch[k] = new_elems
            except Exception as e:
                print(f"[ITP DEBUG] failed to move key {k} to device {device}: {e}")
        return batch

    data_cfg = EasyDict(yaml.safe_load(open(args.data_config)))
    data_module = DataModule(df_path=data_cfg.get('df_path', ''),
                             batch_size=data_cfg.get('batch_size', 1),
                             num_workers=data_cfg.get('num_workers', 2),
                             pin_memory=data_cfg.get('pin_memory', True),
                             cache_dir=data_cfg.get('cache_dir', None),
                             strategy=data_cfg.get('strategy', 'separate'),
                             dataset_args=data_cfg)
    data_module.setup()
    val_loader = data_module.val_dataloader()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_batches = 0
    total_samples = 0
    extracted_samples = 0
    failed_forward = 0
    failed_itp = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            total_batches += 1
            print(f"[ITP DEBUG] Batch {batch_idx}: keys={list(batch.keys())} size={batch.get('size', None)}")
            # move batch tensors to the same device as the model to avoid device mismatch
            try:
                batch = _move_batch_to_device(batch, args.device)
            except Exception as e:
                print(f"[ITP DEBUG] error moving batch to device: {e}")

            try:
                out_embedding, z, _ = model._forward(batch, strategy=batch.get('strategy', 'separate'))
            except Exception as e:
                failed_forward += 1
                print(f"[ITP DEBUG] model._forward failed on batch {batch_idx}: {e}")
                traceback.print_exc()
                # try fallback to regular forward to at least check model invocation
                try:
                    _ = model(batch, batch.get('strategy', 'separate'))
                    print(f"[ITP DEBUG] fallback model.forward succeeded on batch {batch_idx} (no embeddings extracted)")
                except Exception as e2:
                    print(f"[ITP DEBUG] fallback model.forward also failed: {e2}")
                continue

            # compute itp weights from c_former if available
            try:
                itp = model.c_former.itp(out_embedding, z)
            except Exception as e:
                failed_itp += 1
                print(f"[ITP DEBUG] model.c_former.itp failed on batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

            if itp is None:
                failed_itp += 1
                print(f"[ITP DEBUG] model.c_former.itp returned None on batch {batch_idx}")
                continue

            itp = itp.detach().cpu()
            B = itp.shape[0]
            total_samples += B
            for i in range(B):
                sample_id = batch.get('complex', [None] * B)[i] if 'complex' in batch else None
                rows.append({'complex': sample_id, 'w_seq': float(itp[i, 0]), 'w_struct': float(itp[i, 1]), 'w_mix': float(itp[i, 2])})
                extracted_samples += 1

    print(f"[ITP DEBUG] batches={total_batches} total_samples={total_samples} extracted={extracted_samples} failed_forward={failed_forward} failed_itp={failed_itp}")

    # write CSV
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['complex', 'w_seq', 'w_struct', 'w_mix'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Wrote ITP weights to', out_path)


if __name__ == '__main__':
    main()
