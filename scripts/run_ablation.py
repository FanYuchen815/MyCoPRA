#!/usr/bin/env python3
"""Run ablation variants by patching a base model config and invoking run.py.

Usage examples:
  python scripts/run_ablation.py --variant full --model_config config/models/copra.yml --data_config config/datasets/PRA310.yml --run_config config/runs/finetune_struct.yml --output_dir outputs/ablation/full --epochs 50
  python scripts/run_ablation.py --variant "w/o ITP" --pilot --epochs 3
"""
import argparse
import yaml
import subprocess
from pathlib import Path

VARIANTS = {
    'full': {'use_itp': True, 'use_cross': True, 'use_gate': True, 'cross_direction': 'both'},
    'w/o ITP': {'use_itp': False, 'itp_weights': [0.5, 0.3, 0.2], 'use_cross': True, 'use_gate': True},
    'w/o Cross': {'use_itp': True, 'use_cross': False, 'use_gate': True},
    'w/o Gate': {'use_itp': True, 'use_cross': True, 'use_gate': False},
    'uni-directional': {'use_itp': True, 'use_cross': True, 'use_gate': True, 'cross_direction': 'seq2struct'},
    'seq-only': {'mode': 'seq-only'},
    'struct-only': {'mode': 'struct-only'},
}


def patch_model_config(base_model_cfg, variant_cfg):
    cfg = dict(base_model_cfg)
    if 'model' not in cfg:
        raise ValueError('Invalid model config')
    if 'fusion' not in cfg['model']:
        cfg['model']['fusion'] = {}
    fusion = dict(cfg['model']['fusion'])
    # ensure using SSDN enhanced
    fusion['type'] = 'ssdn_enhanced'
    fusion['use_enhanced'] = True
    # apply variant flags
    for k, v in variant_cfg.items():
        fusion[k] = v
    cfg['model']['fusion'] = fusion
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', type=str, required=True)
    p.add_argument('--model_config', type=str, default='config/models/copra.yml')
    p.add_argument('--data_config', type=str, default='config/datasets/PRA310.yml')
    p.add_argument('--run_config', type=str, default='config/runs/finetune_struct.yml')
    p.add_argument('--output_dir', type=str, default='outputs/ablation')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    variant = args.variant
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}. Known: {list(VARIANTS.keys())}')

    base_model_cfg = yaml.safe_load(open(args.model_config))
    base_run_cfg = yaml.safe_load(open(args.run_config))

    out_dir = Path(args.output_dir) / variant.replace(' ', '_')
    out_dir.mkdir(parents=True, exist_ok=True)

    # patch model config
    if 'mode' in VARIANTS[variant] and VARIANTS[variant]['mode'] in ('seq-only', 'struct-only'):
        # for seq-only / struct-only we expect separate baseline registered models
        # The run command will use a different model_type in the model config
        patched_model_cfg = dict(base_model_cfg)
        if VARIANTS[variant]['mode'] == 'seq-only':
            patched_model_cfg['model']['model_type'] = 'sequence_only'
        else:
            patched_model_cfg['model']['model_type'] = 'structure_only'
    else:
        patched_model_cfg = patch_model_config(base_model_cfg, VARIANTS[variant])

    model_cfg_path = out_dir / 'model_config.yml'
    run_cfg_path = out_dir / 'run_config.yml'
    yaml.safe_dump(patched_model_cfg, open(model_cfg_path, 'w'))

    # patch run config if pilot or epochs override
    patched_run_cfg = dict(base_run_cfg)
    if args.epochs is not None:
        patched_run_cfg['epochs'] = int(args.epochs)
    if args.pilot:
        # reduce training time and val frequency for pilot
        patched_run_cfg['epochs'] = int(args.epochs) if args.epochs is not None else 3
        patched_run_cfg['limit_train_batches'] = 0.1
        patched_run_cfg['limit_val_batches'] = 0.2
    yaml.safe_dump(patched_run_cfg, open(run_cfg_path, 'w'))

    cmd = [
        'python', 'run.py', 'finetune', 'dG',
        '--model_config', str(model_cfg_path),
        '--data_config', args.data_config,
        '--run_config', str(run_cfg_path),
    ]

    print('Running:', ' '.join(cmd))
    if args.dry_run:
        return
    subprocess.check_call(cmd)


if __name__ == '__main__':
    main()
