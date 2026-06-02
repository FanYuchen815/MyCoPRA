#!/usr/bin/env python3
"""Compute ITP weights for the full dataset (no train/val/test split).

Generates a CSV with columns: complex,w_seq,w_struct,w_mix

Usage:
  python downstreamTasks/compute_itp_stats_full.py --ckpt <ckpt> --model_config config/models/prorna_ssdn.yml --data_config config/datasets/finetune.yml --out outputs/itp_debug_full.csv --device cpu
"""
import os
import sys
import yaml
import csv
import traceback
from pathlib import Path
import importlib

sys.path.insert(0, os.getcwd())

from easydict import EasyDict
import torch
from torch.utils.data import DataLoader

from data.register import DataRegister


def load_model_from_cfg_and_ckpt(model_cfg_path, ckpt_path, device='cpu'):
    model_cfg = yaml.safe_load(open(model_cfg_path))
    model_args = model_cfg.get('model', {})
    # Ensure model classes are registered
    try:
        importlib.import_module('models.model')
    except Exception:
        pass
    from models.register import ModelRegister
    register = ModelRegister()
    model_type = model_args.get('model_type')
    if model_type is None:
        raise ValueError('model_type not found in model config')
    model_cls = register[model_type]
    model = model_cls(**model_args)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and ('state_dict' in ckpt or 'model' in ckpt):
        state = ckpt.get('state_dict', ckpt.get('model', ckpt))
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            try:
                model.load_state_dict(ckpt, strict=False)
            except Exception:
                pass
    else:
        try:
            model.load_state_dict(ckpt, strict=False)
        except Exception:
            if hasattr(ckpt, 'state_dict'):
                model.load_state_dict(ckpt.state_dict(), strict=False)
    model.to(device)
    model.eval()
    return model


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--model_config', default='config/models/prorna_ssdn.yml')
    p.add_argument('--data_config', default='config/datasets/finetune.yml')
    p.add_argument('--out', default='outputs/itp_debug_full.csv')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=1)
    args = p.parse_args()

    # load model
    model = load_model_from_cfg_and_ckpt(args.model_config, args.ckpt, device=args.device)

    # load data config and full dataframe
    data_cfg = EasyDict(yaml.safe_load(open(args.data_config)))
    df_path = data_cfg.get('df_path')
    if not df_path or not os.path.exists(df_path):
        raise FileNotFoundError(f'df_path not found: {df_path}')
    import pandas as pd
    df = pd.read_csv(df_path)

    # prepare dataset class
    dataset_type = data_cfg.get('dataset_type', 'structure_dataset')
    R = DataRegister()
    if dataset_type not in R:
        raise KeyError(f'Dataset type {dataset_type} not registered')
    dataset_cls = R[dataset_type]

    # Build dataset args from data_cfg
    ds_args = dict(data_cfg)
    # remove df_path and dataset_type not accepted by dataset constructor
    ds_args.pop('df_path', None)
    ds_args.pop('dataset_type', None)

    # instantiate dataset with full dataframe
    dataset = dataset_cls(df, **ds_args)

    # collate function
    try:
        from data.structure_dataset import CustomStructCollate
        collate = CustomStructCollate(strategy=data_cfg.get('strategy', 'separate'))
    except Exception:
        collate = None

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            try:
                # move tensors to device
                dev = torch.device(args.device)
                for k, v in list(batch.items()):
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(dev)
                out_embedding, z, _ = model._forward(batch, strategy=batch.get('strategy', 'separate'))
            except Exception as e:
                print(f'[ITP FULL] model._forward failed on batch {batch_idx}: {e}')
                traceback.print_exc()
                continue

            try:
                itp = model.c_former.itp(out_embedding, z)
            except Exception as e:
                print(f'[ITP FULL] model.c_former.itp failed on batch {batch_idx}: {e}')
                traceback.print_exc()
                continue

            itp = itp.detach().cpu()
            B = itp.shape[0]
            for i in range(B):
                sample_id = batch.get('complex', [None] * B)[i] if 'complex' in batch else None
                rows.append({'complex': sample_id, 'w_seq': float(itp[i, 0]), 'w_struct': float(itp[i, 1]), 'w_mix': float(itp[i, 2])})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['complex', 'w_seq', 'w_struct', 'w_mix'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Wrote ITP weights to', out_path)


if __name__ == '__main__':
    main()
