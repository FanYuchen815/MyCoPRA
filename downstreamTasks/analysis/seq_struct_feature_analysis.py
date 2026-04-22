#!/usr/bin/env python3
"""序列和结构特征贡献度分析

分析模型中：
1. 序列编码器分支 (ESM + RiNALMo) 的贡献
2. 结构编码器分支 (pair_encoder) 的贡献

使用方法：
    python downstreamTasks/analysis/seq_struct_feature_analysis.py \
        --ckpt outputs/finetune_from_pretrained/PRA310/log_fold_0/checkpoint/epoch=63-val_all_pearson=0.000.ckpt \
        --df datasets/PRA310/splits/PRA310.csv \
        --output outputs/seq_struct_importance.csv \
        --device cuda:0
"""
import os
import sys
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
    p.add_argument('--output', default='outputs/seq_struct_importance.csv')
    p.add_argument('--device', default='cpu')
    p.add_argument('--num-samples', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=4)
    return p.parse_args()


class SeqStructAnalyzer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        self.seq_embedding = None
        self.struct_embedding = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """注册hook获取序列和结构embedding"""
        
        def get_seq_embedding(module, input, output):
            if isinstance(output, dict) and 'representation' in output:
                self.seq_embedding = output['representation'].detach()
        
        def get_prot_embedding(module, input, output):
            if isinstance(output, dict):
                for k, v in output.items():
                    if 'representations' in k or 'embedding' in k:
                        self.seq_embedding = v.detach()
                        break
        
        try:
            if hasattr(self.model, 'rinalmo'):
                self.model.rinalmo.register_forward_hook(get_seq_embedding)
        except:
            pass
        
        try:
            if hasattr(self.model, 'esm'):
                self.model.esm.register_forward_hook(get_prot_embedding)
        except:
            pass
        
        try:
            if hasattr(self.model, 'pair_encoder'):
                def get_struct_hook(module, input, output):
                    self.struct_embedding = output.detach()
                self.model.pair_encoder.register_forward_hook(get_struct_hook)
        except:
            pass
    
    def compute_ablation_contribution(self, batch, method='zero'):
        """消融实验：置零序列或结构特征，观察loss变化"""
        batch_orig = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items()}
        
        labels = batch_orig.get('labels')
        if labels is None:
            return None
        
        with torch.no_grad():
            pred_original = self.model(batch_orig, 'separate')
            baseline_loss = F.mse_loss(pred_original.squeeze(), labels.to(self.device).float())
        
        batch_seq_ablated = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch_orig.items()}
        
        batch_struct_ablated = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                               for k, v in batch_orig.items()}
        
        results = {
            'baseline_loss': float(baseline_loss),
            'seq_loss': None,
            'struct_loss': None,
            'both_ablated_loss': None
        }
        
        return results
    
    def compute_gradient_contribution(self, batch):
        """使用梯度方法分析序列和结构特征的贡献"""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        labels = batch.get('labels')
        if labels is None:
            return None
        
        self.model.zero_grad()
        
        labels.requires_grad = True
        pred = self.model(batch, 'separate')
        
        loss = F.mse_loss(pred.squeeze(), labels.to(self.device).float())
        loss.backward()
        
        contribution = {}
        
        if labels.grad is not None:
            grad_magnitude = labels.grad.abs()
            contribution['seq_importance'] = float(grad_magnitude.mean())
            contribution['struct_importance'] = float(grad_magnitude.mean())
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.abs().mean()
                
                if 'esm' in name.lower():
                    contribution.setdefault('esm_grad', []).append(float(grad_norm))
                elif 'rinalmo' in name.lower():
                    contribution.setdefault('rinalmo_grad', []).append(float(grad_norm))
                elif 'pair_encoder' in name.lower() or 'struct' in name.lower():
                    contribution.setdefault('struct_grad', []).append(float(grad_norm))
                elif 'c_former' in name.lower() or 'ssdn' in name.lower():
                    contribution.setdefault('fusion_grad', []).append(float(grad_norm))
        
        for k in ['esm_grad', 'rinalmo_grad', 'struct_grad', 'fusion_grad']:
            if k in contribution and contribution[k]:
                contribution[k] = np.mean(contribution[k])
            else:
                contribution[k] = 0.0
        
        return contribution
    
    def compute_activation_contribution(self, batch):
        """基于激活值大小的贡献分析"""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        self.model.zero_grad()
        
        try:
            self.model(batch, 'separate')
        except:
            pass
        
        contributions = {}
        
        if self.seq_embedding is not None:
            contributions['seq_activation_mean'] = float(self.seq_embedding.abs().mean())
            contributions['seq_activation_max'] = float(self.seq_embedding.abs().max())
        
        if self.struct_embedding is not None:
            contributions['struct_activation_mean'] = float(self.struct_embedding.abs().mean())
            contributions['struct_activation_max'] = float(self.struct_embedding.abs().max())
        
        return contributions


def load_model(args):
    import yaml
    from easydict import EasyDict
    from pl_modules.model_module import ModelModule
    
    print(f"Loading checkpoint: {args.ckpt}")
    module = ModelModule.load_from_checkpoint(args.ckpt, map_location=args.device)
    model = module.model.to(args.device)
    model.eval()
    
    return model


def load_data(args):
    from easydict import EasyDict
    from pl_modules.data_module import DataModule
    import pandas as pd
    
    print(f"Loading data from: {args.df}")
    
    dataset_args = EasyDict({
        'dataset_type': args.dataset_type,
        'col_protein': 'Protein sequences',
        'col_na': 'RNA sequences',
        'col_label': '△G(kcal/mol)'
    })
    
    data_module = DataModule(
        df_path=args.df,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        dataset_args=dataset_args,
        strategy='separate'
    )
    data_module.setup()
    
    return data_module


def main():
    args = parse_args()
    
    print("="*60)
    print("Sequence vs Structure Feature Contribution Analysis")
    print("="*60)
    
    print("\n[1/3] Loading model...")
    model = load_model(args)
    
    print("[2/3] Loading data...")
    data_module = load_data(args)
    val_loader = data_module.val_dataloader()
    
    print("[3/3] Analyzing contributions...")
    
    analyzer = SeqStructAnalyzer(model, args.device)
    
    gradient_results = []
    activation_results = []
    
    num_processed = 0
    for batch in tqdm(val_loader, desc="Processing"):
        if num_processed >= args.num_samples:
            break
        
        try:
            grad_contrib = analyzer.compute_gradient_contribution(batch)
            if grad_contrib:
                gradient_results.append(grad_contrib)
            
            act_contrib = analyzer.compute_activation_contribution(batch)
            if act_contrib:
                activation_results.append(act_contrib)
            
            num_processed += 1
        except Exception as e:
            print(f"\nError on batch {num_processed}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    if gradient_results:
        grad_avg = {}
        for k in gradient_results[0].keys():
            values = [r[k] for r in gradient_results if k in r]
            grad_avg[k] = np.mean(values) if values else 0.0
        
        print("\n--- Gradient-based Contribution ---")
        print(f"{'Component':<25}{'Gradient Magnitude':<20}")
        print("-"*45)
        
        for k in ['esm_grad', 'rinalmo_grad', 'struct_grad', 'fusion_grad']:
            if k in grad_avg:
                print(f"{k:<25}{grad_avg[k]:<20.6f}")
        
        seq_total = grad_avg.get('esm_grad', 0) + grad_avg.get('rinalmo_grad', 0)
        struct_total = grad_avg.get('struct_grad', 0)
        fusion_total = grad_avg.get('fusion_grad', 0)
        
        print(f"\n{'='*45}")
        print(f"SEQUENCE Encoder (ESM+RiNALMo): {seq_total:.6f}")
        print(f"STRUCTURE Encoder (PairEncoder): {struct_total:.6f}")
        print(f"FUSION Module (SSDNEnhanced):     {fusion_total:.6f}")
        print(f"{'='*45}")
        
        total = seq_total + struct_total + fusion_total
        if total > 0:
            print(f"\nRelative Contribution:")
            print(f"  Sequence: {seq_total/total*100:.1f}%")
            print(f"  Structure: {struct_total/total*100:.1f}%")
            print(f"  Fusion: {fusion_total/total*100:.1f}%")
    
    if activation_results:
        act_avg = {}
        for k in activation_results[0].keys():
            values = [r[k] for r in activation_results if k in r]
            act_avg[k] = np.mean(values) if values else 0.0
        
        print("\n--- Activation-based Analysis ---")
        print(f"{'Type':<25}{'Mean':<15}{'Max':<15}")
        print("-"*55)
        for k in ['seq_activation_mean', 'seq_activation_max', 
                  'struct_activation_mean', 'struct_activation_max']:
            if k in act_avg:
                print(f"{k:<25}{act_avg[k]:<15.4f}")
    
    import csv
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['analysis_type', 'component', 'value'])
        
        if gradient_results:
            for k, v in grad_avg.items():
                writer.writerow(['gradient', k, v])
        
        if activation_results:
            for k, v in act_avg.items():
                writer.writerow(['activation', k, v])
    
    print(f"\nResults saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
