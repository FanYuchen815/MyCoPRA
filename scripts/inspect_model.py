import yaml
from easydict import EasyDict
from pathlib import Path
import torch
from models.register import ModelRegister
import importlib
import traceback

# ensure model implementations are imported
try:
    importlib.import_module('models.model')
except Exception:
    pass

cfg = yaml.safe_load(open('config/models/copra.yml'))
model_cfg = EasyDict(cfg)
print('Loaded model cfg keys:', model_cfg.keys())
try:
    R = ModelRegister()
    cls = R[model_cfg['model_type']]
    print('Found model class:', cls)
    # instantiate with kwargs
    m = cls(**model_cfg)
    print('Model instantiated')
    # inspect c_former
    c = getattr(m, 'c_former', None)
    print('c_former type:', type(c))
    if c is not None:
        print('SSDN params:')
        try:
            print('layers:', len(c.layers) if hasattr(c, 'layers') else 'no layers')
            # try to inspect first EntanglementAttention pair dim
            first = c.layers[0]
            print('First layer types:', type(first))
            # check attributes
            print('embed dim:', getattr(first, 'seq_norm').normalized_shape)
            print('struct dim:', getattr(first, 'struct_norm').normalized_shape)
        except Exception as e:
            print('Inspect failed:', e)
except Exception:
    traceback.print_exc()
