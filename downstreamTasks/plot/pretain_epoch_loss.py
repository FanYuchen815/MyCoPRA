#!/usr/bin/env python3
"""根据预训练日志 CSV 绘制 epoch vs 平均 train_loss 曲线。

用法:
    python downstreamTasks/plot/pretain_epoch_loss.py \
        --csv outputs/PRORNA_SSDN_pretrain_enhanced/log_fold_0/lightning_logs/version_2/metrics.csv \
        --out outputs/PRORNA_SSDN_pretrain_enhanced/plots/pretrain_epoch_loss.png
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Plot pretrain epoch loss')
    parser.add_argument('--csv', type=str, default='outputs/PRORNA_SSDN_pretrain_enhanced/log_fold_0/lightning_logs/version_2/metrics.csv')
    parser.add_argument('--out', type=str, default='outputs/PRORNA_SSDN_pretrain_enhanced/plots/pretrain_epoch_loss.png')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')
    args = parser.parse_args()

    try:
        import pandas as pd
    except Exception as e:
        print('缺少 pandas:', e, file=sys.stderr)
        print('请安装：pip install pandas', file=sys.stderr)
        sys.exit(2)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print('缺少 matplotlib:', e, file=sys.stderr)
        print('请安装：pip install matplotlib', file=sys.stderr)
        sys.exit(2)

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f'CSV 文件不存在: {csv_path}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    required_cols = ['train_clip_loss', 'train_dist_loss', 'train_loss']
    available = [c for c in required_cols if c in df.columns]
    if not available:
        print('CSV 中不包含可绘制的 loss 列，至少需要下列之一: ' + ','.join(required_cols), file=sys.stderr)
        print('现有列: ' + ','.join(df.columns), file=sys.stderr)
        sys.exit(1)

    # 按 epoch 分组并取平均（对缺失列跳过）
    grouped = df.groupby('epoch', as_index=True)[available].mean()

    # 确保输出目录
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9,6))
    colors = {'train_clip_loss':'#1f77b4','train_dist_loss':'#ff7f0e','train_loss':'#2ca02c'}
    labels = {'train_clip_loss':'clip loss','train_dist_loss':'dist loss','train_loss':'total loss'}

    for col in available:
        plt.plot(grouped.index.values, grouped[col].values, marker='o', linestyle='-', label=labels.get(col,col), color=colors.get(col,None))

    plt.xlabel('Epoch')
    plt.ylabel('Average loss')
    plt.title('Pretrain: epoch vs average losses')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print('Saved plot to', args.out)

    if args.show:
        plt.show()

if __name__ == '__main__':
    main()
