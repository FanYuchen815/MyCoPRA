#!/usr/bin/env python3
"""ITP 聚类与可解释性分析

读取 `outputs/itp_debug.csv`，对 `w_seq/w_struct/w_mix` 做标准化、PCA、KMeans 聚类，
生成聚类可视化（PCA 散点、权重箱线、ΔG 分布、簇计数）并保存聚类分配与摘要到 `outputs/`。
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main(itp_csv='outputs/itp_debug.csv', meta_csv='datasets/PRA310/splits/PRA310.csv', out_dir='outputs', min_k=2, max_k=6):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(itp_csv):
        print(f"ITP CSV not found: {itp_csv}")
        sys.exit(1)

    df_itp = pd.read_csv(itp_csv)
    # ensure expected columns
    for c in ['complex', 'w_seq', 'w_struct', 'w_mix']:
        if c not in df_itp.columns:
            raise RuntimeError(f"Missing column {c} in {itp_csv}")

    # try load metadata (delta G) if available
    delta_col = None
    if os.path.exists(meta_csv):
        df_meta = pd.read_csv(meta_csv)
        # normalize column name
        meta_cols = {c: c for c in df_meta.columns}
        # prefer '△G(kcal/mol)'
        if '△G(kcal/mol)' in df_meta.columns:
            df_meta['delta_g'] = pd.to_numeric(df_meta['△G(kcal/mol)'], errors='coerce')
            delta_col = 'delta_g'
        else:
            # try to find a numeric column containing 'G' or 'dG'
            found = None
            for c in df_meta.columns:
                if 'G' in c or 'g' in c or 'dG' in c:
                    try:
                        df_meta[c].astype(float)
                        found = c
                        break
                    except Exception:
                        continue
            if found is not None:
                df_meta['delta_g'] = pd.to_numeric(df_meta[found], errors='coerce')
                delta_col = 'delta_g'

        if 'PDB' in df_meta.columns:
            df_meta = df_meta.rename(columns={'PDB': 'complex'})
        df = df_itp.merge(df_meta[['complex', 'delta_g']].drop_duplicates(), on='complex', how='left')
    else:
        df = df_itp.copy()
        df['delta_g'] = np.nan

    # features
    X = df[['w_seq', 'w_struct', 'w_mix']].fillna(0).values
    n_samples = X.shape[0]
    if n_samples < 2:
        print("Not enough samples for clustering")
        sys.exit(1)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # choose k by silhouette (try range)
    best_k = None
    best_score = -1.0
    scores = {}
    max_k = min(max_k, max(2, n_samples - 1))
    for k in range(min_k, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            labels = kmeans.fit_predict(Xs)
            if len(set(labels)) > 1:
                score = silhouette_score(Xs, labels)
            else:
                score = float('nan')
            scores[k] = score
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            scores[k] = float('nan')

    if best_k is None:
        best_k = 3 if n_samples >= 3 else 2

    # final clustering
    km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
    labels = km.fit_predict(Xs)
    df['cluster'] = labels

    # PCA for visualization
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)
    df['pca1'] = pcs[:, 0]
    df['pca2'] = pcs[:, 1]

    # Build a consistent palette mapping for clusters (same mapping used across plots)
    clusters_sorted = sorted(df['cluster'].dropna().unique())
    palette_list = sns.color_palette('tab10', n_colors=max(3, len(clusters_sorted)))
    cluster_palette = {c: palette_list[i % len(palette_list)] for i, c in enumerate(clusters_sorted)}

    sns.set(style='whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette=cluster_palette, ax=ax, s=60)
    ax.set_title(f'PCA of ITP weights (k={best_k})')
    ax.legend(title='cluster')

    ax = axes[0, 1]
    dfm = df.melt(id_vars=['complex', 'cluster'], value_vars=['w_seq', 'w_struct', 'w_mix'], var_name='weight', value_name='value')
    # Use grouped bar plot (mean ± std) to show weight distributions by cluster
    sns.barplot(data=dfm, x='weight', y='value', hue='cluster', palette=cluster_palette, ax=ax, ci='sd', capsize=0.08)
    ax.set_title('Weight distribution by cluster')
    ax.legend(title='cluster', loc='best', frameon=True)

    ax = axes[1, 0]
    if df['delta_g'].notna().sum() > 0:
        sns.boxplot(data=df, x='cluster', y='delta_g', order=clusters_sorted, palette=[cluster_palette[c] for c in clusters_sorted], ax=ax)
        ax.set_title('ΔG by cluster')
    else:
        ax.text(0.5, 0.5, 'No ΔG available', ha='center', va='center')
        ax.set_axis_off()

    ax = axes[1, 1]
    counts = df['cluster'].value_counts().sort_index()
    sns.barplot(x=counts.index, y=counts.values, palette=[cluster_palette[c] for c in counts.index], ax=ax)
    ax.set_title('Cluster counts')
    ax.set_xlabel('cluster')
    ax.set_ylabel('count')

    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'itp_clusters.png')
    fig_pdf = os.path.join(out_dir, 'itp_clusters.pdf')
    fig.savefig(fig_path, dpi=200)
    fig.savefig(fig_pdf)
    plt.close(fig)

    # summary
    summary = df.groupby('cluster').agg(
        count=('complex', 'size'),
        mean_w_seq=('w_seq', 'mean'),
        std_w_seq=('w_seq', 'std'),
        mean_w_struct=('w_struct', 'mean'),
        std_w_struct=('w_struct', 'std'),
        mean_w_mix=('w_mix', 'mean'),
        std_w_mix=('w_mix', 'std'),
        mean_delta_g=('delta_g', 'mean'),
        std_delta_g=('delta_g', 'std')
    ).reset_index()

    summary_path = os.path.join(out_dir, 'itp_cluster_summary.csv')
    assign_path = os.path.join(out_dir, 'itp_cluster_assignments.csv')
    summary.to_csv(summary_path, index=False)
    df.to_csv(assign_path, index=False)

    print('Saved plots to', fig_path, fig_pdf)
    print('Saved summary to', summary_path)
    print('Saved assignments to', assign_path)
    print('Best k by silhouette:', best_k, 'scores:', scores)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--itp_csv', default='outputs/itp_debug.csv')
    p.add_argument('--meta_csv', default='datasets/PRA310/splits/PRA310.csv')
    p.add_argument('--out_dir', default='outputs')
    p.add_argument('--min_k', type=int, default=2)
    p.add_argument('--max_k', type=int, default=6)
    args = p.parse_args()
    main(args.itp_csv, args.meta_csv, args.out_dir, args.min_k, args.max_k)
