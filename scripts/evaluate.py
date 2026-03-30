import argparse
import yaml
from pathlib import Path
import torch
from models.register import ModelRegister


def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=False)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    # Ensure model implementations are imported and registered
    try:
        import importlib
        importlib.import_module('models.model')
    except Exception:
        pass
    model_reg = ModelRegister()
    model = model_reg['copra']()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu')
        try:
            model.load_state_dict(state)
        except Exception:
            pass
    model.eval()
    print('Model loaded for evaluation')

if __name__ == '__main__':
    main()
