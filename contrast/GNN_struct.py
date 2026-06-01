import os
import math
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

import sys
# ensure project root is on sys.path so `data` package can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.complex import ComplexInput


def residue_positions_and_types(complex_input: ComplexInput):
    # compute per-residue centroid from atom41_positions (may contain zeros)
    pos = complex_input.atom41_positions  # (L, 14+27, 3)
    mask = complex_input.atom41_mask
    coords = []
    for i in range(pos.shape[0]):
        m = mask[i]
        if m.sum() == 0:
            coords.append(np.zeros(3, dtype=float))
        else:
            coords.append(pos[i][m.astype(bool)].mean(axis=0))
    coords = np.stack(coords, axis=0)  # (L,3)
    types = complex_input.restype.astype(int)  # integer type per residue
    return coords, types


def build_edge_index(positions, radius=8.0):
    # positions: numpy array (N,3)
    N = positions.shape[0]
    if N == 0:
        return torch.empty((2, 0), dtype=torch.long)
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    src, dst = np.where((dists > 0) & (dists <= radius))
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()


class PRAComplexDataset(Dataset):
    def __init__(self, df, data_root, preload=False):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.samples = []
        for i, row in self.df.iterrows():
            pdb_id = str(row['PDB']).strip()
            # try .pdb then .cif
            p1 = self.data_root / f"{pdb_id}.pdb"
            p2 = self.data_root / f"{pdb_id}.cif"
            if p1.exists():
                path = p1
            elif p2.exists():
                path = p2
            else:
                continue
            self.samples.append((i, path))
        self._cache = None
        if preload:
            self._cache = []
            for i, path in tqdm(self.samples, desc='Preloading PDBs', unit='sample'):
                try:
                    self._cache.append(self._parse_item(i, path))
                except Exception:
                    self._cache.append(None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._cache is not None:
            data = self._cache[idx]
            if data is None:
                raise RuntimeError(f"Failed to parse cached sample idx={idx}")
            return data
        i, path = self.samples[idx]
        return self._parse_item(i, path)

    def _parse_item(self, i, path):
        row = self.df.loc[i]
        comp = ComplexInput.from_path(str(path))
        if comp is None:
            raise RuntimeError(f"Failed to parse {path}")
        pos, types = residue_positions_and_types(comp)
        x = torch.from_numpy(types).long().unsqueeze(1).float()  # simple integer feature
        edge_index = build_edge_index(pos)
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.tensor([float(row['△G(kcal/mol)'])], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y, pos=torch.from_numpy(pos).float())
        return data


class SimpleGNN(nn.Module):
    def __init__(self, in_channels=1, hidden=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch if hasattr(data, 'batch') else None
        x = x.float()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        if batch is None:
            # single graph
            pooled = x.mean(dim=0, keepdim=True)
        else:
            pooled = global_mean_pool(x, batch)
        out = self.head(pooled).squeeze(-1)
        return out


def collate_fn(batch):
    # use torch_geometric Batch utility
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc='Train batches', unit='batch'):
        data = data.to(device)
        opt.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, data.y.view(-1).to(pred.dtype))
        loss.backward()
        opt.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for data in tqdm(loader, desc='Eval batches', unit='batch'):
            data = data.to(device)
            out = model(data)
            preds.append(out.detach().cpu().numpy())
            trues.append(data.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    try:
        from scipy.stats import pearsonr, spearmanr
        pearson = pearsonr(preds, trues)[0] if len(preds) > 1 else float('nan')
        spearman = spearmanr(preds, trues)[0] if len(preds) > 1 else float('nan')
    except Exception:
        pearson = float('nan')
        spearman = float('nan')
    return {'mae': mae, 'rmse': rmse, 'pearson': pearson, 'spearman': spearman}


def run_cv(df_path, pdb_root, epochs=30, batch_size=8, device='cpu', out_dir='./outputs_gnn', preload=False):
    df = pd.read_csv(df_path)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for fold in range(5):
        col = f'fold_{fold}'
        # select rows that have explicit labels for this fold
        train_df = df[df[col] == 'train']
        val_df = df[df[col] == 'val']
        print(f"Fold {fold}: train {len(train_df)} val {len(val_df)}")
        train_ds = PRAComplexDataset(train_df, pdb_root, preload=preload)
        val_ds = PRAComplexDataset(val_df, pdb_root, preload=preload)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = SimpleGNN(in_channels=1).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val = float('inf')
        best_metrics = None
        for ep in range(1, epochs + 1):
            loss = train_one_epoch(model, train_loader, opt, device)
            metrics = evaluate(model, val_loader, device)
            if metrics['mae'] < best_val:
                best_val = metrics['mae']
                best_metrics = metrics
                torch.save(model.state_dict(), os.path.join(out_dir, f'best_fold_{fold}.pt'))
            if ep % 5 == 0 or ep == 1:
                print(f"Fold {fold} Epoch {ep} loss={loss:.4f} val_mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} pearson={metrics['pearson']:.4f}")

        results[f'fold_{fold}'] = best_metrics

    # Aggregate results into a DataFrame and print statistics in requested order
    df_res = pd.DataFrame(results).T
    cols = ['pearson', 'spearman', 'rmse', 'mae']
    # ensure all cols exist
    for c in cols:
        if c not in df_res.columns:
            df_res[c] = float('nan')
    stats = df_res[cols].describe().loc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    # format numeric display
    stats = stats.astype(float).round(6)
    print('\nFinal CV summary:')
    print(stats.to_string())
    # also save to csv
    stats.to_csv(os.path.join(out_dir, 'cv_summary.csv'))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', type=str, default='datasets/PRA310/splits/PRA201.csv')
    parser.add_argument('--pdb_root', type=str, default='datasets/PRA310/PDBs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', type=str, default='./outputs_gnn')
    parser.add_argument('--preload', action='store_true', help='Preload and cache parsed PDB samples before training')
    args = parser.parse_args()
    run_cv(args.df, args.pdb_root, epochs=args.epochs, batch_size=args.batch_size, device=args.device, out_dir=args.out_dir, preload=args.preload)


if __name__ == '__main__':
    main()
