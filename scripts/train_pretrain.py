import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
import utils.safe_checkpoint
from pl_modules.pretune_module import PretuneModule


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--causal', action='store_true', help='Enable causal pretraining losses')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    run_cfg = cfg

    model_cfg_path = run_cfg.get('model_config')

    model_args = {'model': {'model_type': 'PRORNA_SSDN'}}
    data_args = type('X', (), {'batch_size': run_cfg['batch_size'], 'loss_type': 'mse'})
    run_args = run_cfg

    # propagate causal pretrain settings into run_args
    if args.causal:
        run_cfg.setdefault('causal_pretrain', {})
        # allow explicit weight overrides in config
        run_cfg['causal_enabled'] = True

    module = PretuneModule(output_dir=run_cfg.get('output_dir', 'checkpoints/pretrain'), model_args=type('MA', (), {'model': model_args}), data_args=data_args, run_args=run_cfg)

    trainer = pl.Trainer(max_epochs=run_cfg['epochs'], accelerator='gpu' if args.gpus>0 else None, devices=args.gpus if args.gpus>0 else None, precision=run_cfg.get('precision', 32))
    trainer.fit(module)


if __name__ == '__main__':
    main()
