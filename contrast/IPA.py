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
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from tqdm import tqdm

# make repo importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.complex import ComplexInput
import signal


def build_edge_index(positions, radius=8.0):
    pos = positions if isinstance(positions, np.ndarray) else positions.cpu().numpy()
    N = pos.shape[0]
    if N == 0:
        return torch.empty((2, 0), dtype=torch.long)
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    src, dst = np.where((dists > 0) & (dists <= radius))
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index).long()


def residue_centroids(comp: ComplexInput):
    pos41 = comp.atom41_positions
    mask41 = comp.atom41_mask
    coords = []
    for i in range(pos41.shape[0]):
        m = mask41[i]
        if m.sum() == 0:
            coords.append(np.zeros(3, dtype=float))
        else:
            coords.append(pos41[i][m.astype(bool)].mean(axis=0))
    return np.stack(coords, axis=0)


class PRAIPADataset(Dataset):
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
            self.skipped = []
            orig_samples = list(self.samples)
            new_samples = []
            for i, path in tqdm(orig_samples, desc='Preloading', unit='sample'):
                try:
                    # protect parsing with a timeout to avoid hanging on malformed files
                    old_handler = signal.getsignal(signal.SIGALRM)
                    def _raise_timeout(signum, frame):
                        raise TimeoutError('Parsing timed out')
                    signal.signal(signal.SIGALRM, _raise_timeout)
                    signal.alarm(20)  # 20s per file
                    try:
                        data = self._make_data(i, path)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                    if data is None:
                        self.skipped.append((i, str(path), 'parse returned None'))
                    else:
                        self._cache.append(data)
                        new_samples.append((i, path))
                except TimeoutError:
                    self.skipped.append((i, str(path), 'timeout'))
                except Exception as e:
                    self.skipped.append((i, str(path), repr(e)))
            # replace samples with only successfully parsed ones
            self.samples = new_samples
            # save skipped list for debugging
            if len(self.skipped) > 0:
                try:
                    with open('preload_skipped.txt', 'w') as fh:
                        for it in self.skipped:
                            fh.write(f"{it[0]}, {it[1]}, {it[2]}\n")
                except Exception:
                    pass

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
        coords = residue_centroids(comp)
        edge_index = build_edge_index(coords)
        # node features: residue type (one scalar) and mean atom coords
        types = comp.restype.astype(int)
        x_type = torch.from_numpy(types).long().unsqueeze(1).float()
        pos = torch.from_numpy(coords).float()
        x = torch.cat([x_type, pos], dim=1)
        y = torch.tensor([float(row['△G(kcal/mol)'])], dtype=torch.float)
        return Data(x=x, pos=pos, edge_index=edge_index, y=y)


class IPALayer(nn.Module):
    """Simplified Invariant Point Attention-like layer implemented via edge messages."""
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.to_q = nn.Linear(in_dim, hidden)
        self.to_k = nn.Linear(in_dim, hidden)
        self.to_v = nn.Linear(in_dim, hidden)
        # geometric bias MLP: input distance and relative vector
        self.edge_mlp = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.out = nn.Linear(hidden, in_dim)
        # position updater
        self.pos_mlp = nn.Sequential(nn.Linear(hidden, 3), nn.Tanh())

    def forward(self, x, pos, edge_index):
        # x: (N, F), pos: (N,3)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        src = edge_index[0]
        dst = edge_index[1]
        rel = pos[src] - pos[dst]
        dist = torch.norm(rel, dim=-1, keepdim=True)
        geom = torch.cat([dist, rel], dim=-1)
        geo_bias = self.edge_mlp(geom).squeeze(-1)

        # attention score per edge: dot(q_dst, k_src) + geo_bias
        score = (q[dst] * k[src]).sum(dim=-1) + geo_bias
        # normalize per destination node
        alpha = softmax(score, dst)
        # message is v[src] weighted
        msg = v[src] * alpha.unsqueeze(-1)
        agg = scatter_add(msg, dst, dim=0, dim_size=x.size(0))
        x_update = self.out(agg)

        # position update: compute displacement contributions from neighbors
        pos_msg = self.pos_mlp(agg)  # (N,3)
        new_pos = pos + pos_msg
        new_x = x + x_update
        return new_x, new_pos


class IPANet(nn.Module):
    def __init__(self, in_dim, hidden=64, n_layers=3):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([IPALayer(hidden, hidden) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))

    def forward(self, data):
        x = data.x
        pos = data.pos
        edge_index = data.edge_index
        h = self.input_lin(x)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
            h = F.relu(h)
        batch = data.batch if hasattr(data, 'batch') else None
        if batch is None:
            pooled = h.mean(dim=0, keepdim=True)
        else:
            from torch_geometric.nn import global_mean_pool
            pooled = global_mean_pool(h, batch)
        out = self.head(pooled).squeeze(-1)
        return out


def collate_fn(batch):
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def train_epoch(model, loader, opt, device, mean=0.0, std=1.0, normalize=False, clip_grad=None):
    model.train()
    total_loss = 0.0
    for data in tqdm(loader, desc='Train', unit='batch'):
        data = data.to(device)
        opt.zero_grad()
        out = model(data)
        y = data.y.view(-1).to(out.dtype).to(device)
        if normalize:
            y_norm = (y - mean) / (std + 1e-12)
        else:
            y_norm = y
        loss = F.mse_loss(out, y_norm)
        loss.backward()
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, mean=0.0, std=1.0, normalize=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for data in tqdm(loader, desc='Eval', unit='batch'):
            data = data.to(device)
            out = model(data)
            out_np = out.detach().cpu().numpy()
            if normalize:
                out_np = out_np * (std + 1e-12) + mean
            preds.append(out_np)
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


def run_cv(df_path, pdb_root, epochs=30, batch_size=8, device='cpu', out_dir='./outputs_ipa', preload=False):
    df = pd.read_csv(df_path)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for fold in range(5):
        col = f'fold_{fold}'
        train_df = df[df[col] == 'train']
        val_df = df[df[col] == 'val']
        print(f"Fold {fold}: train {len(train_df)} val {len(val_df)}")
        train_ds = PRAIPADataset(train_df, pdb_root, preload=preload)
        val_ds = PRAIPADataset(val_df, pdb_root, preload=preload)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # compute label normalization stats
        train_labels = train_df['△G(kcal/mol)'].astype(float).values
        label_mean = float(train_labels.mean())
        label_std = float(train_labels.std())

        model = IPANet(in_dim=train_ds[0].x.size(1), hidden=64, n_layers=3).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_pcc = -float('inf')
        best_metrics = None
        for ep in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, opt, device, mean=label_mean, std=label_std, normalize=True, clip_grad=1.0)
            metrics = evaluate(model, val_loader, device, mean=label_mean, std=label_std, normalize=True)
            pcc = metrics.get('pearson', float('nan'))
            if not (pcc != pcc):
                if pcc > best_pcc:
                    best_pcc = pcc
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
    print('\nIPA CV summary:')
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
    parser.add_argument('--out_dir', default='./outputs_ipa')
    parser.add_argument('--preload', action='store_true')
    args = parser.parse_args()
    run_cv(args.df, args.pdb_root, epochs=args.epochs, batch_size=args.batch_size, device=args.device, out_dir=args.out_dir, preload=args.preload)


if __name__ == '__main__':
    main()
