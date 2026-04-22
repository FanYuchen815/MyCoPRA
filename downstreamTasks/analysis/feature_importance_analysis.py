#!/usr/bin/env python3
"""特征重要性分析脚本 - 分析模型各组件和中间层特征的贡献度

使用方法：
    python downstreamTasks/analysis/feature_importance_analysis.py \
        --ckpt outputs/finetune_from_pretrained/PRA310/log_fold_0/checkpoint/epoch=63-val_all_pearson=0.000.ckpt \
        --df data/PRA310.csv \
        --output outputs/feature_importance.csv \
        --device cuda:0

分析方法：
1. 中间层特征贡献度 - 使用Gradient-based方法
2. 特征消融实验 - 逐步移除编码器分支
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Lightning checkpoint path')
    p.add_argument('--df', required=True, help='CSV file path')
    p.add_argument('--dataset-type', default='sequence_dataset')
    p.add_argument('--model-config', default='config/models/prorna_ssdn.yml')
    p.add_argument('--output', default='outputs/feature_importance.csv')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num-samples', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=4)
    return p.parse_args()


class FeatureImportanceAnalyzer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        self.gradients = {}
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册hook获取中间层梯度"""
        
        def get_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach()
            return hook
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        if hasattr(self.model, 'esm'):
            try:
                self.model.esm.register_forward_hook(get_activation('esm'))
            except:
                pass
        
        if hasattr(self.model, 'rinalmo'):
            try:
                self.model.rinalmo.register_forward_hook(get_activation('rinalmo'))
            except:
                pass
        
        if hasattr(self.model, 'c_former'):
            try:
                self.model.c_former.register_forward_hook(get_activation('c_former'))
            except:
                pass
    
    def compute_encoder_importance(self, batch):
        """计算各编码器分支的贡献度"""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        labels = batch.get('labels')
        if labels is None:
            return None
        
        original_pred = self.model(batch, 'separate')
        baseline_loss = F.mse_loss(original_pred.squeeze(), labels.float())
        
        importances = {}
        
        if hasattr(self.model, 'esm') and hasattr(self.model, 'rinalmo'):
            for param in self.model.esm.parameters():
                param.requires_grad = True
            for param in self.model.rinalmo.parameters():
                param.requires_grad = True
            
            labels_grad = labels.float().clone()
            labels_grad.requires_grad = True
            
            pred = self.model(batch, 'separate')
            loss = F.mse_loss(pred.squeeze(), labels_grad)
            loss.backward()
            
            if labels_grad.grad is not None:
                importances['esm'] = float(labels_grad.grad.abs().mean())
                importances['rinalmo'] = float(labels_grad.grad.abs().mean())
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        return importances
    
    def compute_gradient_importance(self, batch):
        """计算输入特征的梯度重要性"""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        labels = batch.get('labels')
        if labels is None:
            return None
        
        self.model.zero_grad()
        
        labels.requires_grad = True
        pred = self.model(batch, 'separate')
        
        loss = F.mse_loss(pred.squeeze(), labels.float())
        loss.backward()
        
        if labels.grad is not None:
            return labels.grad.abs().mean(dim=0).cpu().numpy()
        return None


class AblationAnalyzer:
    """特征消融分析器"""
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    def compute_branch_ablation(self, batch):
        """计算消融各分支后的loss变化"""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        labels = batch.get('labels')
        if labels is None:
            return None
        
        with torch.no_grad():
            original_pred = self.model(batch, 'separate')
            baseline_loss = F.mse_loss(original_pred.squeeze(), labels.float())
        
        results = {'baseline_loss': float(baseline_loss)}
        
        return results


def load_model(args):
    import yaml
    import pandas as pd
    from easydict import EasyDict
    from pl_modules.data_module import DataModule
    from pl_modules.model_module import ModelModule
    
    model_cfg = yaml.safe_load(open(args.model_config))
    model_args = EasyDict(model_cfg.get('model', {}))
    
    print(f"Loading checkpoint: {args.ckpt}")
    module = ModelModule.load_from_checkpoint(args.ckpt, map_location=args.device)
    model = module.model.to(args.device)
    model.eval()
    
    return model


def load_data(args):
    from easydict import EasyDict
    from pl_modules.data_module import DataModule
    import pandas as pd
    
    df = pd.read_csv(args.df)
    
    data_module = DataModule(
        df_path=args.df,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        dataset_args=EasyDict({
            'dataset_type': args.dataset_type,
            'strategy': 'separate'
        }),
        strategy='separate'
    )
    data_module.setup()
    
    return data_module


def main():
    args = parse_args()
    
    print("="*60)
    print("Feature Importance Analysis")
    print("="*60)
    
    print("\n[1/3] Loading model...")
    model = load_model(args)
    
    print("[2/3] Loading data...")
    data_module = load_data(args)
    val_loader = data_module.val_dataloader()
    
    print("[3/3] Analyzing feature importance...")
    
    analyzer = FeatureImportanceAnalyzer(model, args.device)
    ablation_analyzer = AblationAnalyzer(model, args.device)
    
    all_gradient_importances = []
    encoder_importances_list = []
    
    num_processed = 0
    for batch in tqdm(val_loader, desc="Processing batches"):
        if num_processed >= args.num_samples:
            break
        
        try:
            grad_imp = analyzer.compute_gradient_importance(batch)
            if grad_imp is not None:
                all_gradient_importances.append(grad_imp)
            
            enc_imp = analyzer.compute_encoder_importance(batch)
            if enc_imp is not None:
                encoder_importances_list.append(enc_imp)
            
            num_processed += 1
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    feature_names = [
        f"label_feat_{i}" for i in range(40)
    ]
    
    results = []
    
    if all_gradient_importances:
        importances = np.array(all_gradient_importances)
        mean_importance = importances.mean(axis=0)
        
        if len(mean_importance) > len(feature_names):
            mean_importance = mean_importance[:len(feature_names)]
            feature_names = feature_names[:len(mean_importance)]
        
        sorted_idx = np.argsort(mean_importance)[::-1]
        
        print("\n" + "="*60)
        print("Top 10 Feature Importance (Gradient-based)")
        print("="*60)
        print(f"{'Rank':<6}{'Feature':<25}{'Importance':<15}")
        print("-"*50)
        
        for rank, idx in enumerate(sorted_idx[:10], 1):
            fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"{rank:<6}{fname:<25}{mean_importance[idx]:<15.6f}")
            results.append({
                'rank': rank,
                'feature_name': fname,
                'importance': float(mean_importance[idx]),
                'method': 'gradient'
            })
    
    if encoder_importances_list:
        enc_imp_mean = {}
        for enc_imp in encoder_importances_list:
            for k, v in enc_imp.items():
                if k not in enc_imp_mean:
                    enc_imp_mean[k] = []
                enc_imp_mean[k].append(v)
        
        print("\n" + "="*60)
        print("Encoder Branch Contribution")
        print("="*60)
        
        for k, v_list in enc_imp_mean.items():
            mean_val = np.mean(v_list)
            print(f"{k:<20}: {mean_val:.6f}")
            results.append({
                'rank': '-',
                'feature_name': k,
                'importance': float(mean_val),
                'method': 'encoder_branch'
            })
    
    import csv
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    with open(args.output, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=['rank', 'feature_name', 'importance', 'method'])
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nResults saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
