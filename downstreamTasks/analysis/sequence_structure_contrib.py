#!/usr/bin/env python3
"""Compare contribution of sequence vs structure features to a regression target.

Usage examples (after extracting features for each layer):
python downstreamTasks/analysis/sequence_structure_contrib.py \
  --features-seq path/to/features_esm.npy \
  --features-struct path/to/features_rinalmo.npy \
  --features-combined path/to/features_cformer.npy \
  --ids path/to/ids.csv \
  --df datasets/PRA310/splits/PRA310.csv \
  --target affinity \
  --outdir downstreamTasks/plot/pic/feat_ana/contrib

The script will train simple regressors (Ridge and RandomForest), report CV R2 scores,
and compute permutation importances to aggregate contribution by modality.
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


def try_load(path):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def map_ids_to_df(ids_df, df):
    # try common column names
    id_col = None
    for cand in ["sample_id", "id", "complex_id", "name", "idx"]:
        if cand in ids_df.columns and cand in df.columns:
            id_col = cand
            break
    if id_col is not None:
        merged = ids_df.merge(df, on=id_col, how="left")
        return merged
    # fallback: assume ids_df first column contains integer indices into df
    firstcol = ids_df.columns[0]
    try:
        idxs = ids_df[firstcol].astype(int).values
        return df.reset_index().loc[idxs].reset_index(drop=True)
    except Exception:
        # as last resort, align by order
        if len(ids_df) != len(df):
            raise ValueError("Cannot align ids with df by order; lengths differ")
        return df.reset_index(drop=True)


def evaluate_regressors(X, y, outdir, prefix):
    results = {}
    if X is None:
        return results
    # simple CV with Ridge and RandomForest
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 7))
    rf = RandomForestRegressor(n_estimators=200, random_state=0)

    # cross-val R2
    ridge_scores = cross_val_score(ridge, X, y, cv=5, scoring="r2")
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    results['ridge_cv_r2_mean'] = float(np.mean(ridge_scores))
    results['rf_cv_r2_mean'] = float(np.mean(rf_scores))

    # train/test split for permutation importance
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    rf.fit(Xtr, ytr)
    ypred = rf.predict(Xte)
    results['rf_r2_test'] = float(r2_score(yte, ypred))
    results['rf_mae_test'] = float(mean_absolute_error(yte, ypred))

    perm = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=0, n_jobs=1)
    results['perm_importances_mean'] = perm.importances_mean.tolist()
    results['perm_importances_std'] = perm.importances_std.tolist()

    # save top importances CSV
    imp_df = pd.DataFrame({
        'feature_idx': np.arange(X.shape[1]),
        'perm_mean': perm.importances_mean,
        'perm_std': perm.importances_std,
    }).sort_values('perm_mean', ascending=False)
    imp_df.to_csv(os.path.join(outdir, f"{prefix}_perm_importances.csv"), index=False)

    # plot top 30
    topk = imp_df.head(30)
    plt.figure(figsize=(6, max(3, len(topk) * 0.2)))
    plt.barh(np.arange(len(topk))[::-1], topk['perm_mean'], xerr=topk['perm_std'])
    plt.yticks(np.arange(len(topk))[::-1], topk['feature_idx'].astype(str))
    plt.xlabel('Permutation importance (mean)')
    plt.title(prefix + ' top permutation importances')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{prefix}_perm_importances.png"), dpi=150)
    plt.close()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-seq', default=None)
    parser.add_argument('--features-struct', default=None)
    parser.add_argument('--features-combined', default=None)
    parser.add_argument('--ids', required=True)
    parser.add_argument('--df', required=True)
    parser.add_argument('--target', default='affinity')
    parser.add_argument('--outdir', default='downstreamTasks/plot/pic/feat_contrib')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    feat_seq = try_load(args.features_seq)
    feat_struct = try_load(args.features_struct)
    feat_comb = try_load(args.features_combined)

    ids_df = pd.read_csv(args.ids)
    df = pd.read_csv(args.df)
    mapped = map_ids_to_df(ids_df, df)

    if args.target not in mapped.columns:
        raise ValueError(f"Target column {args.target} not found in merged dataframe")
    y = mapped[args.target].astype(float).values

    summary = {}
    if feat_seq is not None:
        summary['seq'] = evaluate_regressors(feat_seq, y, args.outdir, 'seq')
    if feat_struct is not None:
        summary['struct'] = evaluate_regressors(feat_struct, y, args.outdir, 'struct')
    if feat_comb is not None:
        summary['combined'] = evaluate_regressors(feat_comb, y, args.outdir, 'combined')

    # If both seq and struct present, aggregate permutation importances to modality-level
    if feat_seq is not None and feat_struct is not None:
        # load the saved perm CSVs if present
        seq_imp = pd.read_csv(os.path.join(args.outdir, 'seq_perm_importances.csv'))
        struct_imp = pd.read_csv(os.path.join(args.outdir, 'struct_perm_importances.csv'))
        seq_total = seq_imp['perm_mean'].sum()
        struct_total = struct_imp['perm_mean'].sum()
        modality_df = pd.DataFrame({'modality': ['sequence', 'structure'], 'perm_sum': [seq_total, struct_total]})
        modality_df.to_csv(os.path.join(args.outdir, 'modality_level_importance.csv'), index=False)
        plt.figure(figsize=(4,3))
        plt.bar(modality_df['modality'], modality_df['perm_sum'])
        plt.ylabel('Sum of permutation importances')
        plt.title('Modality-level contribution')
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'modality_level_importance.png'), dpi=150)
        plt.close()

    # save summary
    pd.Series({k: v for k, v in summary.items()}).to_json(os.path.join(args.outdir, 'contrib_summary.json'))
    print('Done. Results saved to', args.outdir)


if __name__ == '__main__':
    main()
