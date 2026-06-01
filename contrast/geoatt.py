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
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import softmax
from tqdm import tqdm

# ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.complex import ComplexInput


def build_edge_index(positions, radius=8.0):
    # positions: tensor (N,3) or numpy
    pos = positions if isinstance(positions, np.ndarray) else positions.cpu().numpy()
    N = pos.shape[0]
    if N == 0:
        return torch.empty((2, 0), dtype=torch.long)
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    src, dst = np.where((dists > 0) & (dists <= radius))
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()


class PRAComplexStructDataset(Dataset):
    """Dataset exposing only structural information (residue centroids)."""
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
                raise RuntimeError(f'Failed to parse sample idx={idx}')
            return d
        i, path = self.samples[idx]
        return self._make_data(i, path)

    def _make_data(self, i, path):
        row = self.df.loc[i]
        comp = ComplexInput.from_path(str(path))
        if comp is None:
            raise RuntimeError(f'Cannot parse {path}')
        pos = comp.atom41_positions.mean(axis=1)  # average over atom41 positions -> (L,3)
        # fallback if all-zero rows exist
        mask = comp.atom41_mask
        coords = []
        for r_idx in range(pos.shape[0]):
            if mask[r_idx].sum() == 0:
                coords.append(np.zeros(3, dtype=float))
            else:
                coords.append(comp.atom41_positions[r_idx][mask[r_idx].astype(bool)].mean(axis=0))
        coords = np.stack(coords, axis=0)
        edge_index = build_edge_index(coords)
        x = torch.from_numpy(coords).float()  # initial node feature = coordinates
        y = torch.tensor([float(row['△G(kcal/mol)'])], dtype=torch.float)
        return Data(x=x, pos=torch.from_numpy(coords).float(), edge_index=edge_index, y=y)


class GeoAttentionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, hidden=64, radius=8.0):
        super().__init__(aggr='add')
        self.radius = radius
        self.lin_query = nn.Linear(in_channels, hidden)
        self.lin_key = nn.Linear(in_channels, hidden)
        self.lin_value = nn.Linear(in_channels, out_channels)
        # attention MLP on geometric features (distance, direction)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, edge_index, pos):
        # x: (N, in_channels), pos: (N,3)
        q = self.lin_query(x)
        k = self.lin_key(x)
        v = self.lin_value(x)
        return self.propagate(edge_index, x=x, q=q, k=k, v=v, pos=pos)

    def message(self, q_j, k_i, v_j, pos_i, pos_j, index):
        # q_j: query at source? Message signature aligns with propagate args names
        # We'll compute geometric features using pos_i (target) and pos_j (source)
        r = pos_j - pos_i  # (E,3)
        dist = torch.norm(r, dim=-1, keepdim=True)
        edge_feat = torch.cat([dist, r], dim=-1)  # (E,4)
        e = self.edge_mlp(edge_feat).squeeze(-1)  # (E,)
        # attention based on learned geometry and feature similarity
        # compute dot product compatibility
        compat = (q_j * k_i).sum(dim=-1)
        score = e + compat
        alpha = softmax(score, index)
        return v_j * alpha.unsqueeze(-1)


class GeoAttentionNet(nn.Module):
    def __init__(self, in_channels=3, hidden=64, num_layers=3, radius=8.0):
        super().__init__()
        self.input_lin = nn.Linear(in_channels, hidden)
        self.layers = nn.ModuleList([GeoAttentionConv(hidden, hidden, hidden=hidden, radius=radius) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))

    def forward(self, data):
        x = data.x
        pos = data.pos
        edge_index = data.edge_index
        if x is None:
            x = pos
        h = self.input_lin(x)
        for layer in self.layers:
            h = layer(h, edge_index, pos)
            h = F.relu(h)
        # global pooling per graph
        batch = data.batch if hasattr(data, 'batch') else None
        if batch is None:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            pooled = global_mean_pool(h, batch)
        out = self.head(pooled).squeeze(-1)
        return out


def collate_fn(batch):
    return Batch.from_data_list(batch)


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc='Train', unit='batch'):
        data = data.to(device)
        opt.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y.view(-1).to(out.dtype))
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
    from scipy.stats import pearsonr, spearmanr
    pearson = pearsonr(preds, trues)[0] if len(preds) > 1 else float('nan')
    spearman = spearmanr(preds, trues)[0] if len(preds) > 1 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'pearson': pearson, 'spearman': spearman}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', default='datasets/PRA310/splits/PRA201.csv')
    parser.add_argument('--pdb_root', default='datasets/PRA310/PDBs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out_dir', default='./outputs_geo')
    parser.add_argument('--preload', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.df)
    os.makedirs(args.out_dir, exist_ok=True)

    # use fold_0 as quick demo: train on train, val on val
    train_df = df[df['fold_0'] == 'train']
    val_df = df[df['fold_0'] == 'val']
    train_ds = PRAComplexStructDataset(train_df, args.pdb_root, preload=args.preload)
    val_ds = PRAComplexStructDataset(val_df, args.pdb_root, preload=args.preload)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = GeoAttentionNet(in_channels=3, hidden=64, num_layers=3).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = float('inf')
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, args.device)
        metrics = evaluate(model, val_loader, args.device)
        if metrics['mae'] < best:
            best = metrics['mae']
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_geo.pt'))
        print(f'Epoch {ep} loss={loss:.4f} val_mae={metrics["mae"]:.4f} rmse={metrics["rmse"]:.4f} pearson={metrics["pearson"]:.4f} spearman={metrics["spearman"]:.4f}')


if __name__ == '__main__':
    main()
