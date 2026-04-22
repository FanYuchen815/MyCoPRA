#!/usr/bin/env python3
"""统计检验：对聚类结果中的 ΔG 做 Kruskal-Wallis、成对 Mann–Whitney、Spearman 相关。
输出：
- outputs/itp_stat_results.txt
- outputs/itp_pairwise.csv
- outputs/itp_deltaG_by_cluster.png
- outputs/itp_seqbias_vs_deltaG.png
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import itertools
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def main(assign_csv='outputs/itp_cluster_assignments.csv', out_dir='outputs'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(assign_csv):
        print('assignment csv not found:', assign_csv)
        sys.exit(1)

    df = pd.read_csv(assign_csv)
    # ensure numeric
    df['delta_g'] = pd.to_numeric(df.get('delta_g'), errors='coerce')
    df['w_seq'] = pd.to_numeric(df.get('w_seq'), errors='coerce')
    df['w_struct'] = pd.to_numeric(df.get('w_struct'), errors='coerce')
    df['w_mix'] = pd.to_numeric(df.get('w_mix'), errors='coerce')
    df['seq_bias'] = df['w_seq'] - df['w_struct']

    clusters = sorted(df['cluster'].dropna().unique())
    groups = [df.loc[df['cluster'] == c, 'delta_g'].dropna().values for c in clusters]

    # Kruskal-Wallis
    try:
        kw_stat, kw_p = stats.kruskal(*groups)
    except Exception as e:
        kw_stat, kw_p = np.nan, np.nan

    # pairwise Mann-Whitney U
    pair_results = []
    for (i, ci), (j, cj) in itertools.combinations(enumerate(clusters), 2):
        xi = groups[i]
        xj = groups[j]
        if len(xi) < 1 or len(xj) < 1:
            stat, p = np.nan, np.nan
        else:
            try:
                stat, p = stats.mannwhitneyu(xi, xj, alternative='two-sided')
            except Exception:
                stat, p = np.nan, np.nan
        pair_results.append({'cluster_i': ci, 'cluster_j': cj, 'stat': stat, 'p_raw': p})

    # Bonferroni correction
    n_tests = len(pair_results)
    for r in pair_results:
        pr = r['p_raw']
        if pr is None or (isinstance(pr, float) and np.isnan(pr)):
            r['p_bonf'] = np.nan
        else:
            r['p_bonf'] = min(pr * max(1, n_tests), 1.0)

    # Spearman
    valid = df[['seq_bias', 'delta_g']].dropna()
    if len(valid) > 0:
        try:
            sp_r, sp_p = stats.spearmanr(valid['seq_bias'], valid['delta_g'])
        except Exception:
            sp_r, sp_p = np.nan, np.nan
    else:
        sp_r, sp_p = np.nan, np.nan

    # write summary
    with open(os.path.join(out_dir, 'itp_stat_results.txt'), 'w') as f:
        f.write('Kruskal-Wallis for delta_g across clusters\n')
        f.write(f'clusters = {clusters}\n')
        f.write(f'stat = {kw_stat}, p = {kw_p}\n\n')
        f.write('Pairwise Mann-Whitney U (raw p and Bonferroni-corrected)\n')
        for r in pair_results:
            f.write(f"{r['cluster_i']} vs {r['cluster_j']}: stat={r['stat']}, p_raw={r['p_raw']}, p_bonf={r['p_bonf']}\n")
        f.write('\nSpearman between seq_bias and delta_g:\n')
        f.write(f'r = {sp_r}, p = {sp_p}\n')

    # save pairwise csv
    with open(os.path.join(out_dir, 'itp_pairwise.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cluster_i', 'cluster_j', 'stat', 'p_raw', 'p_bonf'])
        writer.writeheader()
        for r in pair_results:
            writer.writerow(r)

    # plots
    sns.set(style='whitegrid')
    plt.figure(figsize=(6, 5))
    clusters_sorted = sorted(df['cluster'].dropna().unique())
    palette_list = sns.color_palette('tab10', n_colors=max(3, len(clusters_sorted)))
    cluster_palette = {c: palette_list[i % len(palette_list)] for i, c in enumerate(clusters_sorted)}
    ax = sns.boxplot(x='cluster', y='delta_g', data=df, order=clusters_sorted, palette=[cluster_palette[c] for c in clusters_sorted])
    ax.set_title('DeltaG by cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'itp_deltaG_by_cluster.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    ax = sns.regplot(x='seq_bias', y='delta_g', data=df, scatter_kws={'s': 40})
    ax.set_title(f'Seq bias vs DeltaG (Spearman r={sp_r:.3f}, p={sp_p:.3e})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'itp_seqbias_vs_deltaG.png'), dpi=200)
    plt.close()

    print('Done. Results in', out_dir)


if __name__ == '__main__':
    main()
