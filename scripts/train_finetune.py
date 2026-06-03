import argparse
import yaml
from pathlib import Path
import pytorch_lightning as pl
import utils.safe_checkpoint
from pl_modules.model_module import ModelModule


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    run_cfg = cfg

    model_cfg_path = run_cfg.get('model_config')

    model_args = {'model_type': 'PRORNA_SSDN'}
    data_args = type('X', (), {'batch_size': run_cfg['batch_size'], 'loss_type': 'mse'})
    run_args = run_cfg

    module = ModelModule(output_dir=run_cfg.get('output_dir', 'checkpoints/finetune'), model_args=type('MA', (), {'model': model_args, 'train': type('T', (), {'task_weights': {'delta_g':1.0,'delta_delta_g':0.5,'binding_site':0.3}})}), data_args=data_args, run_args=run_args)

    trainer = pl.Trainer(max_epochs=run_cfg['epochs'], accelerator='gpu' if args.gpus>0 else None, devices=args.gpus if args.gpus>0 else None, precision=run_cfg.get('precision', 32))
    trainer.fit(module)


if __name__ == '__main__':
    main()
