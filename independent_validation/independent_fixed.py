"""
Independent evaluation script (no training).
Uses StructureDataset + diskcache to avoid re-parsing PDBs.
Uses ModelModule.load_from_checkpoint for correct model weights.
"""
import os
import sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from diskcache import Cache


def spearman_corr(x, y):
    sx = pd.Series(x).rank()
    sy = pd.Series(y).rank()
    return sx.corr(sy)


def pearson_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_model(checkpoint_path, device="cpu"):
    from pl_modules.model_module import ModelModule
    module = ModelModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model = module.model
    model.to(device)
    model.eval()
    epoch = getattr(module, "current_epoch", "?")
    print(f"[INFO] Model restored from Lightning checkpoint (epoch={epoch})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str,
                        default="/root/autodl-tmp/CoPRA/datasets/PRA_single/PDBs")
    parser.add_argument("--csv", type=str,
                        default="/root/autodl-tmp/CoPRA/datasets/PRA_single/PRA_single.csv")
    parser.add_argument("--model", type=str,
                        default="/root/autodl-tmp/CoPRA/outputs/"
                                "finetune_from_pretrained/PRA310/"
                                "epoch=54-val_all_pearson=0.000.ckpt")
    parser.add_argument("--cache-dir", type=str,
                        default="/root/autodl-tmp/CoPRA/cache/pra_single")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    label_col = None
    for c in df.columns:
        if "kcal/mol" in c:
            label_col = c
            break
    if label_col is None:
        for c in df.columns:
            if "label" in c.lower():
                label_col = c
                break
    if label_col is None:
        print("[ERROR] No label column found in CSV")
        return
    print(f"Loaded {len(df)} samples, label column: {label_col}")

    # Phase 1: parse + cache PDBs
    from data.structure_dataset import (
        _process_structure,
        find_case_insensitive_file,
        CustomStructCollate,
    )
    os.makedirs(args.cache_dir, exist_ok=True)
    cache = Cache(args.cache_dir)

    parsed_entries = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Parsing PDBs"):
        pdb = str(row["PDB"]).strip()
        label = row[label_col]
        prot_chains = [s.strip()
                       for s in str(row.get("Protein chains", "")).split(",")
                       if s.strip()]
        na_chains = [s.strip()
                     for s in str(row.get("RNA chains", "")).split(",")
                     if s.strip()]

        if pdb in cache:
            entry = cache[pdb]
            entry["labels"] = float(label)
            entry["complex"] = pdb
            parsed_entries.append(entry)
            continue

        pdb_path = os.path.join(args.data_root, pdb + ".pdb")
        if not os.path.exists(pdb_path):
            pdb_path = os.path.join(args.data_root, pdb + ".cif")
        if not os.path.exists(pdb_path):
            ci = find_case_insensitive_file(args.data_root, pdb + ".pdb")
            if ci is not None:
                pdb_path = ci
        if not os.path.exists(pdb_path):
            print(f"[WARN] missing {pdb}, skipped")
            continue

        proc = _process_structure(
            pdb_path, pdb,
            valid_prot_chains=prot_chains,
            valid_rna_chains=na_chains,
        )
        if proc is None:
            print(f"[WARN] parse failed {pdb}, skipped")
            continue

        L = len(proc["seq"])
        if L > 750:
            print(f"[SKIP] {pdb} has {L} residues ( > 750), skipped")
            continue
        gpu_atoms = proc.get("pos_heavyatom")
        gpu_masks = proc.get("mask_heavyatom")
        try:
            if gpu_atoms is None or gpu_masks is None:
                raise ValueError
            # memory-efficient: loop over atom types instead of 5D broadcast
            n_atoms = gpu_masks.shape[1]
            dist_list = []
            for k in range(n_atoms):
                mask_k = gpu_masks[:, k]  # [L]
                valid = mask_k > 0
                d = torch.full((L, L), float("inf"), dtype=gpu_atoms.dtype, device=gpu_atoms.device)
                if valid.sum() > 1:
                    c = torch.cdist(gpu_atoms[valid, k], gpu_atoms[valid, k])  # [V, V]
                    idx = torch.where(valid)[0]
                    d[idx[:, None], idx[None, :]] = c
                dist_list.append(d)
            atom_min_dist = torch.stack(dist_list, dim=-1).min(dim=-1)[0]
        except Exception:
            atom_min_dist = torch.zeros((L, L))
        try:
            prot_seqs = proc.get("prot_seqs", [])
            max_prot_length = max(len(s) for s in prot_seqs) if prot_seqs else 0
        except Exception:
            max_prot_length = 0
        try:
            na_key = "rna_seqs" if "rna_seqs" in proc else "na_seqs"
            na_seqs = proc.get(na_key, [])
            max_na_length = max(len(s) for s in na_seqs) if na_seqs else 0
        except Exception:
            max_na_length = 0

        proc["atom_min_dist"] = atom_min_dist
        proc["max_prot_length"] = int(max_prot_length)
        proc["max_na_length"] = int(max_na_length)
        proc["labels"] = float(label)
        proc["complex"] = pdb

        cache[pdb] = proc
        parsed_entries.append(proc)

    print(f"Parsed {len(parsed_entries)} structures (cache: {len(cache)} entries)")

    # Phase 2: apply transforms (same as training)
    from data.transforms import get_transform
    transform_cfg = [
        {"type": "select_atom", "resolution": "backbone"},
        {"type": "selected_region_with_distmap", "patch_size": 2048},
        {"type": "subtract_center_of_mass"},
    ]
    transform = get_transform(transform_cfg)

    transformed_data = []
    pdb_ids = []
    truths = []
    for entry in tqdm(parsed_entries, desc="Transforming"):
        try:
            t = transform(entry)
            transformed_data.append(t)
            pdb_ids.append(entry["complex"])
            truths.append(entry["labels"])
        except Exception as e:
            print(f"[WARN] transform failed for {entry.get('complex','?')}: {e}, skipped")

    # Phase 3: inference
    model = load_model(args.model, device=args.device)
    collate = CustomStructCollate()

    preds = []
    n = len(transformed_data)
    for i in tqdm(range(0, n, args.batch_size), desc="Inference"):
        chunk = transformed_data[i : i + args.batch_size]
        batch = collate(chunk)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, dict):
                out = out[list(out.keys())[0]]
            if isinstance(out, tuple):
                out = out[0]
            out = out.cpu().numpy().reshape(-1)
        preds.extend(out.tolist())
    preds = np.array(preds)

    # Phase 4: metrics
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    mae = float(np.mean(np.abs(preds - truths)))
    pcc = pearson_corr(preds, truths)
    scc = spearman_corr(preds, truths)

    print("")
    print("=" * 50)
    print("Evaluation results on PRA_single:")
    print("=" * 50)
    print(f"  PCC:  {pcc:.6f}")
    print(f"  SCC:  {scc:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  N={len(preds)}")
    print("=" * 50)

    out_df = pd.DataFrame({"PDB": pdb_ids, "pred": preds, "true": truths})
    out_path = os.path.join(os.path.dirname(args.csv), "predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
