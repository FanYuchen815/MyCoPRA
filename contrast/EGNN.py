import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
from tqdm import tqdm

# ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.complex import ComplexInput


def residue_centroids_and_types(comp: ComplexInput):
    pos41 = comp.atom41_positions
    mask41 = comp.atom41_mask
    coords = []
    for i in range(pos41.shape[0]):
        m = mask41[i]
        if m.sum() == 0:
            coords.append(np.zeros(3, dtype=float))
        else:
            coords.append(pos41[i][m.astype(bool)].mean(axis=0))
    coords = np.stack(coords, axis=0)
    types = comp.restype.astype(int)
    return coords, types


def build_edge_index(positions, radius=8.0):
    pos = positions if isinstance(positions, np.ndarray) else positions.cpu().numpy()
    N = pos.shape[0]
    if N == 0:
        return torch.empty((2, 0), dtype=torch.long)
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    src, dst = np.where((dists > 0) & (dists <= radius))
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()


class PRAComplexDataset(Dataset):
    def __init__(self, df, pdb_root, preload=False):
        self.df = df.reset_index(drop=True)
        self.pdb_root = Path(pdb_root)
        self.samples = []
        for i, row in self.df.iterrows():
            pdb_id = str(row['PDB']).strip()
            p1 = self.pdb_root / f"{pdb_id}.pdb"
            p2 = self.pdb_root / f"{pdb_id}.cif"
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
            for i, path in tqdm(self.samples, desc='Preloading', unit='sample'):
                try:
                    self._cache.append(self._make_data(i, path))
                except Exception:
                    self._cache.append(None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self._cache is not None:
            d = self._cache[idx]
            if d is None:
                raise RuntimeError(f'Failed to parse cached sample idx={idx}')
            return d
        i, path = self.samples[idx]
        return self._make_data(i, path)

    def _make_data(self, i, path):
        row = self.df.loc[i]
        comp = ComplexInput.from_path(str(path))
        if comp is None:
            raise RuntimeError(f'Cannot parse {path}')
        coords, types = residue_centroids_and_types(comp)
        edge_index = build_edge_index(coords)
        # node feature: residue type (as float) concatenated with coords
        x_type = torch.from_numpy(types).long().unsqueeze(1).float()
        pos = torch.from_numpy(coords).float()
        x = torch.cat([x_type, pos], dim=1)
        y = torch.tensor([float(row['△G(kcal/mol)'])], dtype=torch.float)
        return Data(x=x, pos=pos, edge_index=edge_index, y=y)


class EGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(in_dim * 2 + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.msg_mlp = nn.Linear(hidden, out_dim)
        self.coord_mlp = nn.Linear(hidden, 1)
        self.node_mlp = nn.Sequential(nn.Linear(in_dim + out_dim, out_dim), nn.ReLU())

    def forward(self, x, pos, edge_index):
        # x: (N, F), pos: (N,3), edge_index: (2, E)
        src = edge_index[0]
        dst = edge_index[1]
        r = pos[src] - pos[dst]
        dist2 = (r ** 2).sum(dim=-1, keepdim=True)
        edge_feat = torch.cat([x[src], x[dst], dist2], dim=-1)
        h_e = self.edge_mlp(edge_feat)
        msg = self.msg_mlp(h_e)
        coord_weight = torch.sigmoid(self.coord_mlp(h_e))
        # aggregate messages
        agg = scatter_add(msg, dst, dim=0, dim_size=x.size(0))
        # update node features
        new_x = self.node_mlp(torch.cat([x, agg], dim=-1))
        # coordinate update
        coord_msg = r * coord_weight
        coord_agg = scatter_add(coord_msg, dst, dim=0, dim_size=pos.size(0))
        new_pos = pos + coord_agg
        return new_x, new_pos


class EGNNNet(nn.Module):
    def __init__(self, in_dim, hidden=64, n_layers=4):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([EGNNLayer(hidden, hidden) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, data):
        x = data.x
        pos = data.pos
        edge_index = data.edge_index
        h = self.input_lin(x)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        batch = data.batch if hasattr(data, 'batch') else None
        if batch is None:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            pooled = global_mean_pool(h, batch)
        out = self.head(pooled).squeeze(-1)
        return out


def collate_fn(batch):
    return Batch.from_data_list(batch)


def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc='Train', unit='batch'):
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
    preds, trues = [], []
    with torch.no_grad():
        for data in tqdm(loader, desc='Eval', unit='batch'):
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


def run_cv(df_path, pdb_root, epochs=30, batch_size=8, device='cpu', out_dir='./outputs_egnn', preload=False):
    df = pd.read_csv(df_path)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for fold in range(5):
        col = f'fold_{fold}'
        train_df = df[df[col] == 'train']
        val_df = df[df[col] == 'val']
        print(f"Fold {fold}: train {len(train_df)} val {len(val_df)}")
        train_ds = PRAComplexDataset(train_df, pdb_root, preload=preload)
        val_ds = PRAComplexDataset(val_df, pdb_root, preload=preload)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        model = EGNNNet(in_dim=train_ds[0].x.size(1), hidden=64, n_layers=4).to(device)
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

    df_res = pd.DataFrame(results).T
    cols = ['pearson', 'spearman', 'rmse', 'mae']
    for c in cols:
        if c not in df_res.columns:
            df_res[c] = float('nan')
    stats = df_res[cols].describe().loc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats = stats.astype(float).round(6)
    print('\nEGNN CV summary:')
    print(stats.to_string())
    stats.to_csv(os.path.join(out_dir, 'cv_summary.csv'))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', default='datasets/PRA310/splits/PRA201.csv')
    parser.add_argument('--pdb_root', default='datasets/PRA310/PDBs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', default='./outputs_egnn')
    parser.add_argument('--preload', action='store_true')
    args = parser.parse_args()
    run_cv(args.df, args.pdb_root, epochs=args.epochs, batch_size=args.batch_size, device=args.device, out_dir=args.out_dir, preload=args.preload)


if __name__ == '__main__':
    main()
