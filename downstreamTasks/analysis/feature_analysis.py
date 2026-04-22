#!/usr/bin/env python3
"""特征分析脚本：从 Lightning checkpoint 加载模型，提取验证/测试集的中间特征，做 PCA/UMAP/KMeans 可视化。

示例用法：
  python downstreamTasks/analysis/feature_analysis.py \
    --ckpt outputs/finetune_from_pretrained/PRA310/log_fold_0/checkpoint/epoch=63-val_all_pearson=0.000.ckpt \
    --df data/your_dataset.csv \
    --dataset-type sequence_dataset \
    --outdir downstreamTasks/plot/pic/feat_analysis \
    --device cuda:0 \
    --layer model.encoder

说明：
  - 默认使用 `pl_modules.data_module.DataModule` 来创建 dataloader，需通过 `--dataset-args` 传入 JSON 格式的 dataset 参数（比如列名、dataset_type 等）。
  - `--layer` 可选，若提供（例如 `model.encoder`），脚本将在该子模块上注册 forward hook 并收集中间激活作为特征；否则脚本会使用模型的输出（若为 Tensor 或可转为 numpy）。
"""
import os
import sys
import json
import argparse
import tempfile
import math
from typing import Any

# Ensure repo root is on sys.path so project packages like `pl_modules` can be imported
try:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
except Exception:
    pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Lightning checkpoint path')
    p.add_argument('--df', required=True, help='CSV file used by DataModule (full path relative to repo)')
    p.add_argument('--dataset-type', default='sequence_dataset', help='registered dataset type')
    p.add_argument('--dataset-args', default='{}', help='JSON string of dataset args passed to dataset constructor')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--device', default='cpu')
    p.add_argument('--outdir', default='downstreamTasks/plot/pic/feat_analysis')
    p.add_argument('--layer', default=None, help='dot path to layer to hook (e.g. "model.encoder")')
    p.add_argument('--n-pca', type=int, default=50)
    p.add_argument('--n-clusters', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=None, help='only process up to this many samples (for quick tests)')
    return p.parse_args()


def get_attr_by_path(obj: Any, path: str):
    parts = path.split('.')
    cur = obj
    for p in parts:
        cur = getattr(cur, p)
    return cur


def main():
    args = parse_args()

    try:
        import torch
        import numpy as np
    except Exception as e:
        print('请确保安装 torch 和 numpy:', e, file=sys.stderr)
        sys.exit(2)

    try:
        from pl_modules.data_module import DataModule
        from pl_modules.model_module import ModelModule
    except Exception as e:
        print('无法导入项目模块，请确认在仓库根目录运行脚本:', e, file=sys.stderr)
        sys.exit(2)

    # 解析 dataset_args
    try:
        ds_args = json.loads(args.dataset_args)
    except Exception:
        print('dataset-args 必须为 JSON 字符串', file=sys.stderr)
        sys.exit(1)
    # 确保 dataset_type 在 args
    ds_args['dataset_type'] = args.dataset_type
    # If structure dataset and no transform provided, add a default select_atom transform
    if args.dataset_type == 'structure_dataset' and not ds_args.get('transform'):
        ds_args['transform'] = [
            {
                'type': 'select_atom',
                'resolution': 'backbone'
            }
        ]
        print('Added default transform: select_atom(backbone) to provide pos_atoms/mask_atoms')

    # DataModule expects dataset_args to support attribute access (like EasyDict)
    try:
        from easydict import EasyDict
        ds_args = EasyDict(ds_args)
    except Exception:
        from types import SimpleNamespace
        ds_args = SimpleNamespace(**ds_args)

    # 初始化 DataModule (optionally slice CSV for quick tests)
    df_path = args.df
    tmp_csv_path = None
    if args.max_samples is not None:
        try:
            import pandas as pd
            df = pd.read_csv(args.df)
            df = df.head(int(args.max_samples)).copy()
            tmp_fd, tmp_csv_path = tempfile.mkstemp(prefix='feat_analysis_', suffix='.csv')
            os.close(tmp_fd)
            df.to_csv(tmp_csv_path, index=False)
            df_path = tmp_csv_path
            print(f'Using sliced CSV with first {len(df)} rows:', df_path)
        except Exception as e:
            print('无法裁剪 CSV，继续使用完整数据集:', e, file=sys.stderr)

    dm = DataModule(df_path=df_path, batch_size=args.batch_size, num_workers=args.num_workers, dataset_args=ds_args)
    dm.setup()
    dl = dm.val_dataloader()

    # 加载 LightningModule
    print('加载 checkpoint:', args.ckpt)
    map_loc = args.device
    module = None
    try:
        module = ModelModule.load_from_checkpoint(args.ckpt, map_location=map_loc)
    except Exception as e:
        # 尝试直接使用 torch.load 的 state_dict 加载到 ModelModule() 的新实例（保守）
        print('ModelModule.load_from_checkpoint 失败，尝试直接载入 state_dict:', e)
        try:
            ck = torch.load(args.ckpt, map_location=map_loc)
            model_module = ModelModule(model_args=None, data_args=None)
            state_dict = ck.get('state_dict', ck)
            model_module.load_state_dict(state_dict, strict=False)
            module = model_module
        except Exception as e2:
            print('直接加载 state_dict 也失败:', e2, file=sys.stderr)
            sys.exit(1)

    module.eval()
    model = module.model if hasattr(module, 'model') else module
    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu')
    model.to(device)

    # If running on CPU, replace Flash attention blocks with standard attention to avoid CPU-only errors.
    if device.type == 'cpu':
        try:
            from rinalmo.model.attention import MultiHeadSelfAttention, FlashMultiHeadSelfAttention

            def _replace_flash_attn(mod):
                for name, child in list(mod.named_children()):
                    # recurse first
                    _replace_flash_attn(child)
                    # replace flash attention module with standard attention
                    if isinstance(child, FlashMultiHeadSelfAttention):
                        embed_dim = child.embed_dim
                        num_heads = child.num_heads
                        use_rot_emb = getattr(child, 'use_rot_emb', True)
                        bias = child.Wqkv.bias is not None
                        new_attn = MultiHeadSelfAttention(embed_dim, num_heads, child.attention_dropout.p, use_rot_emb, bias)

                        # copy weights from fused Wqkv into q/k/v projections
                        with torch.no_grad():
                            w = child.Wqkv.weight
                            q_w, k_w, v_w = torch.chunk(w, 3, dim=0)
                            new_attn.mh_attn.to_q.weight.copy_(q_w)
                            new_attn.mh_attn.to_k.weight.copy_(k_w)
                            new_attn.mh_attn.to_v.weight.copy_(v_w)
                            if child.Wqkv.bias is not None:
                                b = child.Wqkv.bias
                                q_b, k_b, v_b = torch.chunk(b, 3, dim=0)
                                new_attn.mh_attn.to_q.bias.copy_(q_b)
                                new_attn.mh_attn.to_k.bias.copy_(k_b)
                                new_attn.mh_attn.to_v.bias.copy_(v_b)
                            # copy output projection
                            new_attn.mh_attn.out_proj.weight.copy_(child.out_proj.weight)
                            if child.out_proj.bias is not None:
                                new_attn.mh_attn.out_proj.bias.copy_(child.out_proj.bias)

                        setattr(mod, name, new_attn)
                    # flip any explicit flash-attn flag on blocks
                    if hasattr(child, 'use_flash_attn'):
                        try:
                            child.use_flash_attn = False
                        except Exception:
                            pass

            _replace_flash_attn(model)
            print('Disabled flash attention for CPU execution (replaced with standard attention)')
        except Exception as e:
            print('警告：无法自动替换 Flash attention（CPU 模式）:', e, file=sys.stderr)

    # Disable flash-attn on CPU to avoid flash_attn rotary backend errors
    if device.type == 'cpu':
        try:
            # Flip any module-level use_flash_attn flags
            for m in model.modules():
                if hasattr(m, 'use_flash_attn'):
                    try:
                        setattr(m, 'use_flash_attn', False)
                    except Exception:
                        pass
            # Also update RiNALMo config flag if present
            if hasattr(model, 'rinalmo') and hasattr(model.rinalmo, 'config'):
                try:
                    model.rinalmo.config.model.transformer.use_flash_attn = False
                except Exception:
                    pass
            print('Flash attention disabled for CPU execution')
        except Exception:
            pass

    features_list = []
    labels_list = []
    ids_list = []

    hook_handle = None
    captured = {'feat': []}

    def _hook(module_hook, input, output):
        if isinstance(output, torch.Tensor):
            captured['feat'].append(output.detach().cpu())
        elif isinstance(output, (list, tuple)):
            captured['feat'].append(output[0].detach().cpu())
        else:
            # try converting
            try:
                t = torch.tensor(output)
                captured['feat'].append(t.detach().cpu())
            except Exception:
                pass

    if args.layer:
        # allow users to pass --layer model.encoder even when `model` is already the inner model
        layer_path = args.layer
        if layer_path.startswith('model.') and not hasattr(model, 'model'):
            layer_path = layer_path[len('model.') :]
        try:
            layer = get_attr_by_path(model, layer_path)
            hook_handle = layer.register_forward_hook(_hook)
            print('Registered hook on', args.layer, '-> resolved to', layer_path)
        except Exception as e:
            # try to help by listing children and doing a fuzzy match
            print('注册 hook 失败:', e, file=sys.stderr)
            try:
                children = list(model.named_children())
                print('Model children:', [name for name, _ in children])
                # try to find a child name that contains the requested leaf
                leaf = layer_path.split('.')[-1]
                candidates = [name for name, _ in children if leaf in name or leaf.replace('_','') in name]
                if candidates:
                    chosen = candidates[0]
                    print(f"尝试用相似子模块 '{chosen}' 注册 hook（候选来自层名匹配）")
                    layer = getattr(model, chosen)
                    hook_handle = layer.register_forward_hook(_hook)
                    print('Registered hook on', chosen)
                else:
                    print('未找到匹配的子模块可供注册 hook，请检查 model 结构', file=sys.stderr)
                    sys.exit(1)
            except Exception as e2:
                print('在尝试自动匹配子模块时出错:', e2, file=sys.stderr)
                sys.exit(1)

    total = 0
    processed_samples = 0
    # If max_samples given, compute conservative max number of batches to iterate
    max_batches = None
    if args.max_samples is not None and args.batch_size and args.batch_size > 0:
        max_batches = int(math.ceil(float(args.max_samples) / float(args.batch_size)))
    for batch_i, batch in enumerate(dl):
        # move tensors in batch to device if tensors
        def to_device(x):
            if torch.is_tensor(x):
                return x.to(device)
            return x
        batch_on_dev = {k: to_device(v) for k, v in batch.items()}

        with torch.no_grad():
            out = None
            try:
                # model may expect (batch, strategy) as in ModelModule
                strategy = batch_on_dev.get('strategy', getattr(dm, 'strategy', None))
                if strategy is not None:
                    out = model(batch_on_dev, strategy)
                else:
                    out = model(batch_on_dev)
            except TypeError:
                # try passing only batch
                out = model(batch_on_dev)

        if args.layer:
            # collect captured feats for this batch
            if len(captured['feat']) == 0:
                print('警告：hook 未捕获到任何输出，检查 layer 路径或模型前向是否触发该子模块', file=sys.stderr)
            else:
                # stack captured list items produced by forward hook (one per forward)
                b_feats = torch.cat(captured['feat'], dim=0) if len(captured['feat']) > 1 else captured['feat'][0]
                features_list.append(b_feats.numpy())
                captured['feat'].clear()
        else:
            # use model output
            if isinstance(out, torch.Tensor):
                feat = out.detach().cpu().numpy()
            elif isinstance(out, (list, tuple)):
                first = out[0]
                feat = first.detach().cpu().numpy() if torch.is_tensor(first) else np.array(first)
            else:
                try:
                    feat = np.array(out)
                except Exception:
                    print('无法解析模型输出为特征，提供 --layer 参数以捕获中间特征', file=sys.stderr)
                    sys.exit(1)
            features_list.append(feat)

        # collect labels if present and robustly infer batch size
        batch_size_here = 1
        if 'labels' in batch:
            lab = batch['labels'].cpu().numpy()
            labels_list.append(lab)
            batch_size_here = int(lab.shape[0])
        else:
            # infer batch size from batch tensors if possible, otherwise from captured feats or outputs
            try:
                found = False
                for v in batch.values():
                    if hasattr(v, 'size') and torch.is_tensor(v):
                        batch_size_here = int(v.size(0))
                        found = True
                        break
                if not found:
                    if args.layer and 'b_feats' in locals():
                        batch_size_here = int(b_feats.shape[0])
                    elif 'feat' in locals():
                        batch_size_here = int(feat.shape[0])
            except Exception:
                batch_size_here = 1

        # collect ids if present
        if 'complex' in batch:
            try:
                ids_list.extend(list(batch['complex']))
            except Exception:
                pass
        elif 'id' in batch:
            try:
                ids_list.extend(list(batch['id']))
            except Exception:
                pass

        processed_samples += int(batch_size_here)
        total += 1
        if total % 10 == 0:
            print(f'Processed {total} batches ({processed_samples} samples)')
        if args.max_samples is not None and processed_samples >= args.max_samples:
            print(f'Reached max-samples={args.max_samples}, stopping early')
            break
        if max_batches is not None and (batch_i + 1) >= max_batches:
            # also stop after estimated number of batches
            print(f'Reached max-batches={max_batches} (from max-samples), stopping early')
            break

    if hook_handle is not None:
        hook_handle.remove()
    if tmp_csv_path and os.path.exists(tmp_csv_path):
        try:
            os.remove(tmp_csv_path)
        except Exception:
            pass

    import numpy as np
    features = np.concatenate([np.asarray(x) for x in features_list], axis=0)
    labels = np.concatenate(labels_list, axis=0) if labels_list else None

    print('Features shape:', features.shape)

    # 简单分析：PCA -> UMAP -> KMeans -> 保存
    try:
        from sklearn.decomposition import PCA
        import umap
        from sklearn.cluster import KMeans
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('请先安装 sklearn, umap-learn, matplotlib 等依赖:', e, file=sys.stderr)
        sys.exit(2)

    os.makedirs(args.outdir, exist_ok=True)

    pca_n = min(args.n_pca, features.shape[0], features.shape[1])
    pca = PCA(n_components=pca_n)
    z = pca.fit_transform(features)
    # UMAP can fail when n_samples is very small; handle small-N gracefully
    if z.shape[0] < 4:
        # fallback to first 2 PCA dims (or pad if needed)
        if z.shape[1] >= 2:
            embedding = z[:, :2]
        else:
            pad = np.zeros((z.shape[0], 2 - z.shape[1]))
            embedding = np.concatenate([z, pad], axis=1)
    else:
        n_neighbors = min(10, z.shape[0] - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        embedding = reducer.fit_transform(z)
    k_clusters = min(args.n_clusters, z.shape[0])
    kmeans = KMeans(n_clusters=k_clusters, random_state=0).fit(z)

    # 保存原始特征
    np.save(os.path.join(args.outdir, 'features.npy'), features)
    np.save(os.path.join(args.outdir, 'embedding.npy'), embedding)
    np.save(os.path.join(args.outdir, 'kmeans_labels.npy'), kmeans.labels_)
    if labels is not None:
        np.save(os.path.join(args.outdir, 'labels.npy'), labels)

    # Save ids if available
    if ids_list:
        try:
            import pandas as pd
            ids_df = pd.DataFrame({
                'index': list(range(len(ids_list))),
                'sample_id': ids_list
            })
            if labels is not None and len(labels) == len(ids_list):
                ids_df['label'] = labels
            ids_df.to_csv(os.path.join(args.outdir, 'ids.csv'), index=False)
        except Exception:
            pass

    # 绘图
    plt.figure(figsize=(6,5))
    if labels is not None:
        sc = plt.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='tab10', s=6)
        cbar = plt.colorbar(sc)
        cbar.set_label('标签')
    else:
        sc = plt.scatter(embedding[:,0], embedding[:,1], c=kmeans.labels_, cmap='tab10', s=6)
        cbar = plt.colorbar(sc)
        cbar.set_label('KMeans簇')
    plt.title('UMAP特征可视化')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'umap.png'), dpi=150)

    # 聚类分布
    plt.figure(figsize=(6,5))
    plt.hist(kmeans.labels_, bins=np.arange(k_clusters+1)-0.5)
    plt.title('KMeans簇分布')
    plt.xlabel('簇ID')
    plt.ylabel('样本数')
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'kmeans_hist.png'), dpi=150)

    # 特征TopK占比（PCA解释方差累积）
    try:
        var_ratio = pca.explained_variance_ratio_
        cum_ratio = np.cumsum(var_ratio)
        k = np.arange(1, len(cum_ratio) + 1)
        plt.figure(figsize=(6,5))
        plt.plot(k, cum_ratio, marker='o', markersize=3, linewidth=1)
        plt.ylim(0, 1.02)
        plt.title('TopK特征累计占比(PCA)')
        plt.xlabel('K(主成分数量)')
        plt.ylabel('累计解释方差占比')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'topk_pca_cumratio.png'), dpi=150)
        np.save(os.path.join(args.outdir, 'pca_explained_variance_ratio.npy'), var_ratio)
    except Exception:
        pass

    print('Analysis results saved to', args.outdir)


if __name__ == '__main__':
    main()
