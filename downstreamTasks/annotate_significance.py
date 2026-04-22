#!/usr/bin/env python3
"""在 ΔG 箱线图上标注成对显著性（使用 outputs/itp_pairwise.csv）。

生成：
- outputs/itp_deltaG_by_cluster_annotated.png/.pdf
"""
import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def p_to_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'n.s.'
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def main(assign_csv='outputs/itp_cluster_assignments.csv', pair_csv='outputs/itp_pairwise.csv', out_png='outputs/itp_deltaG_by_cluster_annotated.png'):
    if not os.path.exists(assign_csv):
        print('Missing', assign_csv); sys.exit(1)
    if not os.path.exists(pair_csv):
        print('Missing', pair_csv); sys.exit(1)

    df = pd.read_csv(assign_csv)
    pair = pd.read_csv(pair_csv)

    # ensure p_bonf exists
    if 'p_bonf' not in pair.columns:
        n = len(pair)
        pair['p_bonf'] = pair['p_raw'] * max(1, n)
        pair['p_bonf'] = pair['p_bonf'].clip(upper=1.0)

    # prepare plot
    sns.set(style='whitegrid')
    plt.figure(figsize=(6, 5))
    order = sorted(df['cluster'].unique())
    # consistent palette mapping
    palette_list = sns.color_palette('tab10', n_colors=max(3, len(order)))
    cluster_palette = {c: palette_list[i % len(palette_list)] for i, c in enumerate(order)}
    ax = sns.boxplot(x='cluster', y='delta_g', data=df, order=order, palette=[cluster_palette[c] for c in order])
    ax.set_title('ΔG by cluster (with significance)')

    # mapping cluster label -> x position
    cluster_to_x = {c: i for i, c in enumerate(order)}

    # select significant pairs (p_bonf <= 0.05)
    sig_pairs = []
    for _, r in pair.iterrows():
        ci = r['cluster_i']
        cj = r['cluster_j']
        try:
            ci = int(ci); cj = int(cj)
        except Exception:
            pass
        p = r.get('p_bonf', r.get('p_raw', np.nan))
        try:
            p = float(p)
        except Exception:
            p = np.nan
        if not math.isnan(p) and p <= 0.05:
            if ci in cluster_to_x and cj in cluster_to_x:
                sig_pairs.append((ci, cj, p))

    # sort pairs by span (wider spans higher) so labels don't overlap too much
    sig_pairs = sorted(sig_pairs, key=lambda x: abs(cluster_to_x[x[0]] - cluster_to_x[x[1]]), reverse=True)

    # compute baseline y and step
    y_max = df['delta_g'].max() if not df['delta_g'].isnull().all() else 0
    y_min = df['delta_g'].min() if not df['delta_g'].isnull().all() else 0
    span = y_max - y_min if (y_max - y_min) != 0 else 1.0
    step = span * 0.06

    used_levels = []
    for idx, (ci, cj, p) in enumerate(sig_pairs):
        x1 = cluster_to_x[ci]
        x2 = cluster_to_x[cj]
        if x1 == x2:
            continue
        left = min(x1, x2)
        right = max(x1, x2)

        # choose a y level that is not used (avoid overlap)
        level = 1
        while True:
            y = y_max + step * level
            conflict = any(abs(y - h) < (step * 0.5) for h in used_levels)
            if not conflict:
                used_levels.append(y)
                break
            level += 1

        # draw annotation line
        ax.plot([left, left, right, right], [y - step*0.12, y, y, y - step*0.12], lw=1.5, color='k')
        stars = p_to_stars(p)
        ax.text((left + right)/2.0, y + step*0.02, f'{stars} (p={p:.2g})', ha='center', va='bottom', color='k', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_png.replace('.png', '.pdf'))
    plt.close()
    print('Saved annotated plot to', out_png)


if __name__ == '__main__':
    main()
