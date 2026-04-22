# Feature-Dimension Correlation

This script computes per-dimension correlations between extracted features and dataset attributes (affinity, sequence length, GC content).

## Inputs
- `features.npy`: extracted features (N x D)
- `ids.csv`: mapping of feature rows to sample IDs (optional but recommended)
- dataset CSV: contains affinity and sequence columns

## Usage
```bash
python downstreamTasks/analysis/feature_dim_correlation.py \
  --features downstreamTasks/plot/pic/feat_analysis_cformer/features.npy \
  --ids downstreamTasks/plot/pic/feat_analysis_cformer/ids.csv \
  --csv datasets/PRA310/splits/PRA310.csv \
  --affinity-col '△G(kcal/mol)' \
  --outdir downstreamTasks/plot/pic/feature_correlation \
  --topk 20
```

## Outputs
- `corr_affinity.csv`, `corr_prot_len.csv`, `corr_rna_len.csv`, `corr_rna_gc.csv`
- `topk_affinity.png`, `topk_prot_len.png`, `topk_rna_len.png`, `topk_rna_gc.png`

## Dependencies
- numpy, pandas, scipy, matplotlib

If needed:
```bash
pip install scipy
```
