#!/usr/bin/env python3
"""Compute per-dimension correlations between features and sequence/affinity attributes."""
import argparse
import os
import sys
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True, help='Path to features.npy (N x D)')
    p.add_argument('--ids', default=None, help='Path to ids.csv (optional, from feature_analysis.py)')
    p.add_argument('--csv', required=True, help='Original dataset CSV (with affinity and sequences)')
    p.add_argument('--id-col', default='PDB', help='ID column in CSV')
    p.add_argument('--affinity-col', default='△G(kcal/mol)', help='Affinity/label column in CSV')
    p.add_argument('--outdir', default='downstreamTasks/plot/pic/feature_correlation')
    p.add_argument('--topk', type=int, default=20, help='Top K dims by |correlation| to plot')
    return p.parse_args()


def _split_seqs(cell: str) -> List[str]:
    if cell is None:
        return []
    parts = str(cell).split(',')
    seqs = []
    for p in parts:
        p = p.strip()
        if ':' in p:
            p = p.split(':', 1)[1]
        if p:
            seqs.append(p)
    return seqs


def _gc_fraction(seq: str) -> float:
    if not seq:
        return float('nan')
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / max(len(seq), 1)


def main():
    args = parse_args()
    try:
        import numpy as np
        import pandas as pd
        from scipy.stats import pearsonr, spearmanr
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('Missing dependencies: numpy, pandas, scipy, matplotlib', e, file=sys.stderr)
        sys.exit(2)

    feats = np.load(args.features)
    if feats.ndim != 2:
        print('features.npy must be 2D (N x D)', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)

    # Build mapping from ids.csv if provided
    if args.ids and os.path.exists(args.ids):
        ids_df = pd.read_csv(args.ids)
        if 'sample_id' not in ids_df.columns:
            print('ids.csv must contain sample_id column', file=sys.stderr)
            sys.exit(1)
        ids_df['sample_id_str'] = ids_df['sample_id'].astype(str)
        ids_df['sample_id_base'] = ids_df['sample_id_str'].str.split('_').str[0]
        df_id = df.copy()
        df_id[args.id_col] = df_id[args.id_col].astype(str)
        merged = ids_df.merge(df_id, left_on='sample_id_str', right_on=args.id_col, how='left')
        # fallback to base id
        if merged[args.id_col].isna().any():
            merged2 = ids_df.merge(df_id, left_on='sample_id_base', right_on=args.id_col, how='left')
            # keep rows where merged is missing
            merged.loc[merged[args.id_col].isna(), df_id.columns] = merged2.loc[merged[args.id_col].isna(), df_id.columns].values
        df_use = merged
    else:
        df_use = df.copy()
        if len(df_use) != feats.shape[0]:
            print('ids.csv missing and CSV length does not match features rows.', file=sys.stderr)
            sys.exit(1)

    # sequence-derived attributes
    prot_seqs_col = 'Protein sequences' if 'Protein sequences' in df_use.columns else None
    rna_seqs_col = 'RNA sequences' if 'RNA sequences' in df_use.columns else None

    prot_lens = []
    rna_lens = []
    rna_gc = []

    for _, row in df_use.iterrows():
        prot_seqs = _split_seqs(row[prot_seqs_col]) if prot_seqs_col else []
        rna_seqs = _split_seqs(row[rna_seqs_col]) if rna_seqs_col else []
        prot_len = sum(len(s) for s in prot_seqs) if prot_seqs else float('nan')
        rna_len = sum(len(s) for s in rna_seqs) if rna_seqs else float('nan')
        gc_vals = [_gc_fraction(s) for s in rna_seqs] if rna_seqs else []
        gc_val = float('nan') if not gc_vals else sum(gc_vals) / len(gc_vals)
        prot_lens.append(prot_len)
        rna_lens.append(rna_len)
        rna_gc.append(gc_val)

    df_use['prot_len'] = prot_lens
    df_use['rna_len'] = rna_lens
    df_use['rna_gc'] = rna_gc

    targets = {
        'affinity': df_use.get(args.affinity_col, None),
        'prot_len': df_use['prot_len'],
        'rna_len': df_use['rna_len'],
        'rna_gc': df_use['rna_gc'],
    }

    os.makedirs(args.outdir, exist_ok=True)

    for tname, tvals in targets.items():
        if tvals is None:
            continue
        t = np.asarray(tvals, dtype=float)
        valid = np.isfinite(t)
        if valid.sum() < 3:
            continue
        X = feats[valid]
        y = t[valid]

        records = []
        for dim in range(X.shape[1]):
            x = X[:, dim]
            if np.allclose(x, x[0]):
                continue
            try:
                pr, pp = pearsonr(x, y)
            except Exception:
                pr, pp = float('nan'), float('nan')
            try:
                sr, sp = spearmanr(x, y)
            except Exception:
                sr, sp = float('nan'), float('nan')
            records.append((dim, pr, pp, sr, sp))

        out_csv = os.path.join(args.outdir, f'corr_{tname}.csv')
        import pandas as pd
        out_df = pd.DataFrame(records, columns=['dim', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p'])
        out_df['abs_pearson'] = out_df['pearson_r'].abs()
        out_df = out_df.sort_values('abs_pearson', ascending=False)
        out_df.to_csv(out_csv, index=False)

        # plot top-k
        topk = out_df.head(args.topk)
        if len(topk) > 0:
            plt.figure(figsize=(8, 4))
            plt.bar([str(d) for d in topk['dim']], topk['abs_pearson'])
            plt.title(f'TopK特征相关性 |{tname}|')
            plt.xlabel('特征维度')
            plt.ylabel(' |Pearson r|')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, f'topk_{tname}.png'), dpi=150)

    print('Saved correlation outputs to', args.outdir)


if __name__ == '__main__':
    main()
