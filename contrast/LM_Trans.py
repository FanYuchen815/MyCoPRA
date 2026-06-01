"""
LM + Transformer baseline for RNA-protein binding affinity prediction.
LOCAL MODEL ONLY VERSION
NO CACHE
5-FOLD CV

Outputs:
- Pearson
- Spearman
- RMSE
- MAE
"""

import os
import json
import warnings
import gc
import shutil
import traceback
import random

import pandas as pd
import numpy as np
import esm

from tqdm import tqdm

import torch
import torch.nn as nn
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# Disable cache
# ============================================================
os.environ["TRANSFORMERS_CACHE"] = ""
os.environ["HF_HOME"] = ""
os.environ["TORCH_HOME"] = ""
os.environ["HF_DATASETS_CACHE"] = ""

# ============================================================
# Config
# ============================================================
ESM_DIM = 1280
RNA_DIM = 1280

HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 4

DROPOUT = 0.1

LR = 2e-4
WEIGHT_DECAY = 1e-5

EPOCHS = 5

BATCH_SIZE = 4
ACCUM_STEPS = 4

MAX_PROT_LEN = 1022
MAX_RNA_LEN = 512

PATIENCE = 10

DATA_DIR = "datasets"

PRA310_PATH = f"{DATA_DIR}/PRA310/splits/PRA310.csv"
PRA201_PATH = f"{DATA_DIR}/PRA310/splits/PRA201.csv"

OUTPUT_DIR = "contrast/outputs"

CACHE_DIR = f"{DATA_DIR}/cache/lm_trans"

# ============================================================
# Local model paths
# ============================================================
ESM_LOCAL_PATH = (
    "/root/autodl-tmp/CoPRA/weights/esm2_t33_650M_UR50D.pt"
)

RINALMO_LOCAL_PATH = (
    "/root/autodl-tmp/CoPRA/weights/rinalmo_giga_pretrained.pt"
)

# ============================================================
# Remove old cache
# ============================================================
if os.path.exists(CACHE_DIR):

    print(f"Removing cache: {CACHE_DIR}")

    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")

# -----------------------
# Reproducibility / seed
# -----------------------
SEED = 42

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# PyTorch deterministic options (may slow down)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set CUBLAS workspace config to improve determinism on CUDA >=10.2
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# global generator for DataLoader shuffles
_GLOBAL_DL_GENERATOR = torch.Generator()
_GLOBAL_DL_GENERATOR.manual_seed(SEED)

# ============================================================
# Fold columns
# ============================================================
fold_cols = [
    "fold_0",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
]

# ============================================================
# Helpers
# ============================================================
def parse_seq(seq_str, key):

    if pd.isna(seq_str):
        return ""

    parts = [p.strip() for p in str(seq_str).split(",")]

    # priority key
    for part in parts:

        if part.startswith(key + ":"):

            return part.split(":", 1)[1]

    # fallback
    for part in parts:

        if ":" in part:

            return part.split(":", 1)[1]

    return ""


def build_dataset(
    df,
    prot_key="A",
    rna_key="B",
    label_col="△G(kcal/mol)"
):

    items = []

    for idx, row in df.iterrows():

        try:

            prot = parse_seq(
                row["Protein sequences"],
                prot_key
            )

            rna = parse_seq(
                row["RNA sequences"],
                rna_key
            )

            if prot is None:
                prot = ""

            if rna is None:
                rna = ""

            label = np.nan

            if (
                label_col in row
                and pd.notna(row[label_col])
            ):
                label = float(row[label_col])

            items.append(
                (prot, rna, label)
            )

        except Exception as e:

            print(
                f"build_dataset error row={idx}: {e}"
            )

            items.append(
                ("", "", np.nan)
            )

    return items


# ============================================================
# Dataset
# ============================================================
class SeqPairDataset(Dataset):

    def __init__(self, items):

        self.items = items

    def __len__(self):

        return len(self.items)

    def __getitem__(self, idx):

        prot, rna, label = self.items[idx]

        return prot, rna, label


# ============================================================
# Model
# ============================================================
class LMTransformer(nn.Module):

    def __init__(
        self,
        esm_dim=1280,
        rna_dim=1280,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ):

        super().__init__()

        self.proj_prot = nn.Sequential(
            nn.LayerNorm(esm_dim),
            nn.Linear(esm_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.proj_rna = nn.Sequential(
            nn.LayerNorm(rna_dim),
            nn.Linear(rna_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        max_len = MAX_PROT_LEN + MAX_RNA_LEN + 3

        self.pos_embed = nn.Embedding(
            max_len,
            hidden_dim
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):

        for p in self.parameters():

            if p.dim() > 1:

                nn.init.xavier_uniform_(p)

    def forward(
        self,
        prot_emb,
        rna_emb,
        prot_len,
        rna_len
    ):

        B = prot_emb.size(0)

        prot = self.proj_prot(prot_emb)

        rna = self.proj_rna(rna_emb)

        cls_tok = self.cls_token.expand(B, -1, -1)

        x = torch.cat(
            [cls_tok, prot, rna],
            dim=1
        )

        pos_ids = (
            torch.arange(
                x.size(1),
                device=x.device
            )
            .unsqueeze(0)
            .expand(B, -1)
        )

        x = x + self.pos_embed(pos_ids)

        mask = torch.zeros(
            B,
            x.size(1),
            dtype=torch.bool,
            device=x.device
        )

        for i in range(B):

            valid_len = (
                1
                + int(prot_len[i])
                + int(rna_len[i])
            )

            valid_len = min(
                valid_len,
                x.size(1)
            )

            mask[i, :valid_len] = True

        x = self.transformer(
            x,
            src_key_padding_mask=~mask
        )

        cls_out = x[:, 0, :]

        pred = self.head(cls_out).squeeze(-1)

        return pred


# ============================================================
# ESM extraction
# ============================================================
def extract_esm_embeddings(
    seqs,
    batch_converter,
    model,
    batch_size=2,
    max_len=1022
):

    model.eval()

    all_embs = []

    for i in range(0, len(seqs), batch_size):

        batch_seqs = seqs[i:i + batch_size]

        batch_seqs = [
            str(s)[:max_len]
            if s is not None else ""
            for s in batch_seqs
        ]

        try:

            batch_data = [
                (f"seq_{j}", s if s != "" else "A")
                for j, s in enumerate(batch_seqs)
            ]

            _, _, batch_tokens = batch_converter(
                batch_data
            )

            batch_tokens = batch_tokens.to(DEVICE)

            with torch.no_grad():

                results = model(
                    batch_tokens,
                    repr_layers=[33],
                    return_contacts=False
                )

                token_embs = (
                    results["representations"][33]
                    .cpu()
                )

            for b_idx, seq in enumerate(batch_seqs):

                valid_len = min(
                    len(seq) + 1,
                    token_embs.shape[1]
                )

                if valid_len <= 1:

                    emb = torch.zeros(1, ESM_DIM)

                else:

                    emb = token_embs[
                        b_idx,
                        1:valid_len
                    ]

                all_embs.append(emb)

        except Exception as e:

            print(f"ESM extraction error: {e}")

            traceback.print_exc()

            for _ in range(len(batch_seqs)):

                all_embs.append(
                    torch.zeros(1, ESM_DIM)
                )

    return all_embs


# ============================================================
# RiNALMo extraction
# ============================================================
def extract_rinalmo_embeddings(
    seqs,
    model,
    tokenizer,
    batch_size=2,
    max_len=512
):

    model.eval()

    all_embs = []

    for i in range(0, len(seqs), batch_size):

        batch_seqs = seqs[i:i + batch_size]

        batch_seqs = [
            str(s)[:max_len].replace("T", "U")
            if s is not None else "A"
            for s in batch_seqs
        ]

        try:

            token_ids = tokenizer.batch_tokenize(
                batch_seqs
            )

            input_ids = torch.tensor(
                token_ids,
                dtype=torch.long
            ).to(DEVICE)

            with torch.no_grad():

                outputs = model(input_ids)

                emb = outputs["representation"].cpu()

            for b_idx, seq in enumerate(batch_seqs):

                mask = (
                    torch.tensor(token_ids[b_idx])
                    != tokenizer.pad_idx
                )

                valid = int(mask.sum().item())

                if valid > 2:

                    emb_t = emb[
                        b_idx,
                        1:valid - 1
                    ]

                else:

                    emb_t = torch.zeros(1, RNA_DIM)

                all_embs.append(emb_t)

        except Exception as e:

            print(f"RiNALMo extraction error: {e}")

            traceback.print_exc()

            for _ in range(len(batch_seqs)):

                all_embs.append(
                    torch.zeros(1, RNA_DIM)
                )

    return all_embs


# ============================================================
# Collate
# ============================================================
def collate_emb(batch):

    prot_embs = []
    rna_embs = []

    prot_lens = []
    rna_lens = []

    labels = []

    for prot_e, rna_e, label in batch:

        prot_embs.append(prot_e)
        rna_embs.append(rna_e)

        prot_lens.append(prot_e.size(0))
        rna_lens.append(rna_e.size(0))

        labels.append(
            0.0 if np.isnan(label) else label
        )

    max_prot = max(prot_lens)
    max_rna = max(rna_lens)

    B = len(prot_embs)

    prot_pad = torch.zeros(
        B,
        max_prot,
        prot_embs[0].size(-1)
    )

    rna_pad = torch.zeros(
        B,
        max_rna,
        rna_embs[0].size(-1)
    )

    for i in range(B):

        prot_pad[i, :prot_lens[i]] = prot_embs[i]

        rna_pad[i, :rna_lens[i]] = rna_embs[i]

    return (
        prot_pad.float(),
        rna_pad.float(),
        torch.tensor(prot_lens),
        torch.tensor(rna_lens),
        torch.tensor(labels, dtype=torch.float),
    )


# ============================================================
# Trainer
# ============================================================
class Trainer:

    def __init__(self, model, device):

        self.model = model.to(device)

        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )

        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=EPOCHS
            )
        )

        self.criterion = nn.MSELoss()

    def train_epoch(self, loader):

        self.model.train()

        total_loss = 0
        n = 0

        self.optimizer.zero_grad()

        for step, (
            prot_emb,
            rna_emb,
            prot_len,
            rna_len,
            labels
        ) in enumerate(
            tqdm(loader, desc="Train", leave=False)
        ):

            prot_emb = prot_emb.to(self.device)
            rna_emb = rna_emb.to(self.device)

            labels = labels.to(self.device)

            prot_len = prot_len.to(self.device)
            rna_len = rna_len.to(self.device)

            pred = self.model(
                prot_emb,
                rna_emb,
                prot_len,
                rna_len
            )

            loss = self.criterion(pred, labels)

            loss = loss / ACCUM_STEPS

            loss.backward()

            if (
                (step + 1) % ACCUM_STEPS == 0
                or (step + 1) == len(loader)
            ):

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += (
                loss.item()
                * len(labels)
                * ACCUM_STEPS
            )

            n += len(labels)

        return total_loss / n

    @torch.no_grad()
    def evaluate(self, loader):

        self.model.eval()

        all_preds = []
        all_labels = []

        for (
            prot_emb,
            rna_emb,
            prot_len,
            rna_len,
            labels
        ) in tqdm(loader, desc="Eval", leave=False):

            prot_emb = prot_emb.to(self.device)
            rna_emb = rna_emb.to(self.device)

            labels = labels.to(self.device)

            prot_len = prot_len.to(self.device)
            rna_len = rna_len.to(self.device)

            pred = self.model(
                prot_emb,
                rna_emb,
                prot_len,
                rna_len
            )

            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())

        preds = torch.cat(all_preds).numpy()

        labels = torch.cat(all_labels).numpy()

        rmse = np.sqrt(
            mean_squared_error(labels, preds)
        )

        mae = mean_absolute_error(labels, preds)

        pearson, _ = pearsonr(labels, preds)

        spearman, _ = spearmanr(labels, preds)

        return {
            "pearson": pearson,
            "spearman": spearman,
            "rmse": rmse,
            "mae": mae,
        }


# ============================================================
# Main
# ============================================================
def main():

    print("=" * 60)
    print("LM + Transformer Baseline")
    print("=" * 60)

    # ========================================================
    # Load data
    # ========================================================
    # Use PRA201 as the main dataset for 5-fold CV, and PRA310 as the external test set
    df310 = pd.read_csv(PRA201_PATH)

    df201 = pd.read_csv(PRA310_PATH)

    print(f"df310 rows = {len(df310)}")
    print(f"df201 rows = {len(df201)}")

    print("\nColumns:")
    print(df310.columns.tolist())

    label_col = [
        c
        for c in df310.columns
        if "G" in c and "kcal" in c
    ][0]

    print(f"\nLabel column: {label_col}")

    all_items = build_dataset(
        df310,
        label_col=label_col
    )

    test_items = build_dataset(
        df201,
        label_col=label_col
    )

    print(f"\nall_items  = {len(all_items)}")
    print(f"test_items = {len(test_items)}")

    prot_all = [x[0] for x in all_items]
    rna_all = [x[1] for x in all_items]

    prot_test = [x[0] for x in test_items]
    rna_test = [x[1] for x in test_items]

    # ========================================================
    # Load ESM local
    # ========================================================
    print("\n[1/4] Loading ESM local")

    assert os.path.exists(ESM_LOCAL_PATH)

    try:

        esm_model, esm_alphabet = (
            esm.pretrained.load_model_and_alphabet_local(
                ESM_LOCAL_PATH
            )
        )

    except Exception as e:

        print(f"Fallback loading: {e}")

        model_data = torch.load(
            ESM_LOCAL_PATH,
            map_location="cpu"
        )

        model_name = Path(
            ESM_LOCAL_PATH
        ).stem

        esm_model, esm_alphabet = (
            esm.pretrained.load_model_and_alphabet_core(
                model_name,
                model_data,
                None,
            )
        )

    esm_model = esm_model.eval().to(DEVICE)

    esm_batch_converter = (
        esm_alphabet.get_batch_converter()
    )

    print("ESM loaded")

    # ========================================================
    # Load RiNALMo local
    # ========================================================
    print("\n[2/4] Loading RiNALMo local")

    from rinalmo.config import model_config
    from rinalmo.model.model import RiNALMo
    from rinalmo.data.alphabet import Alphabet

    cfg = model_config("giga")

    rna_model = RiNALMo(cfg)

    state = torch.load(
        RINALMO_LOCAL_PATH,
        map_location="cpu"
    )

    if isinstance(state, dict):

        if "model" in state:
            state = state["model"]

        if "state_dict" in state:
            state = state["state_dict"]

    rna_model.load_state_dict(state)

    rna_model = rna_model.eval().to(DEVICE)

    rna_tokenizer = Alphabet()

    print("RiNALMo loaded")

    # ========================================================
    # Extract embeddings
    # ========================================================
    print("\n[3/4] Extracting embeddings")

    prot_embs_all = extract_esm_embeddings(
        prot_all,
        esm_batch_converter,
        esm_model
    )

    rna_embs_all = extract_rinalmo_embeddings(
        rna_all,
        rna_model,
        rna_tokenizer
    )

    prot_embs_test = extract_esm_embeddings(
        prot_test,
        esm_batch_converter,
        esm_model
    )

    rna_embs_test = extract_rinalmo_embeddings(
        rna_test,
        rna_model,
        rna_tokenizer
    )

    print("\nEmbedding diagnostics")

    print(
        f"len(prot_embs_all) = {len(prot_embs_all)}"
    )

    print(
        f"len(rna_embs_all)  = {len(rna_embs_all)}"
    )

    assert len(prot_embs_all) == len(df310)
    assert len(rna_embs_all) == len(df310)

    # ========================================================
    # Free GPU memory
    # ========================================================
    del esm_model
    del rna_model

    gc.collect()

    torch.cuda.empty_cache()

    # ========================================================
    # Prepare datasets
    # ========================================================
    embs_train_val = list(
        zip(
            prot_embs_all,
            rna_embs_all,
            [x[2] for x in all_items]
        )
    )

    embs_test = list(
        zip(
            prot_embs_test,
            rna_embs_test,
            [x[2] for x in test_items]
        )
    )

    test_dataset = SeqPairDataset(
        embs_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        collate_fn=collate_emb,
        num_workers=0,
        worker_init_fn=worker_init_fn,
        generator=_GLOBAL_DL_GENERATOR,
    )

    # ========================================================
    # 5-fold CV
    # ========================================================
    print("\n[4/4] Running 5-fold CV")

    cv_results = []

    for fold_idx in range(5):

        print("\n" + "=" * 50)

        print(f"Fold {fold_idx + 1}/5")

        print("=" * 50)

        # For each fold: use CSV's 'val' label as the fold's test set.
        # We do NOT use a separate validation set here — train on 'train',
        # then evaluate on the fold 'val' rows as the test set.
        train_mask = (
            df310[fold_cols[fold_idx]] == "train"
        )

        test_mask = (
            df310[fold_cols[fold_idx]] == "val"
        )

        print(
            f"train={int(train_mask.sum())} "
            f"test={int(test_mask.sum())}"
        )

        train_embs = [
            embs_train_val[i]
            for i in range(len(all_items))
            if train_mask.iloc[i]
        ]

        test_embs_fold = [
            embs_train_val[i]
            for i in range(len(all_items))
            if test_mask.iloc[i]
        ]

        print(
            f"selected train={len(train_embs)} "
            f"selected test={len(test_embs_fold)}"
        )

        train_dataset = SeqPairDataset(train_embs)

        test_dataset_fold = SeqPairDataset(test_embs_fold)

        # per-fold generators to keep shuffle deterministic per fold
        g_train = torch.Generator()
        g_train.manual_seed(SEED + fold_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_emb,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            generator=g_train,
        )

        test_loader_fold = DataLoader(
            test_dataset_fold,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_emb,
            num_workers=0,
            worker_init_fn=worker_init_fn,
            generator=_GLOBAL_DL_GENERATOR,
        )

        model = LMTransformer()

        trainer = Trainer(model, DEVICE)

        for epoch in range(1, EPOCHS + 1):

            train_loss = trainer.train_epoch(train_loader)

            if epoch == 1 or epoch % 10 == 0:

                print(
                    f"Epoch {epoch:02d} | "
                    f"Train={train_loss:.4f}"
                )

            trainer.scheduler.step()

        # After fixed training, evaluate on the fold's test set
        best_test_metrics = trainer.evaluate(test_loader_fold)

        # Save final model for the fold
        torch.save(
            model.state_dict(),
            f"{OUTPUT_DIR}/fold{fold_idx}_final.pt"
        )

        print("\nFold Test Metrics")

        print(best_test_metrics)

        cv_results.append(best_test_metrics)

        del model
        del trainer

        gc.collect()

        torch.cuda.empty_cache()

    # ========================================================
    # Summary
    # ========================================================
    print("\n" + "=" * 60)

    print("5-Fold CV Results Summary")

    print("=" * 60)

    summary_df = pd.DataFrame({

        "pearson": [
            r["pearson"]
            for r in cv_results
        ],

        "spearman": [
            r["spearman"]
            for r in cv_results
        ],

        "rmse": [
            r["rmse"]
            for r in cv_results
        ],

        "mae": [
            r["mae"]
            for r in cv_results
        ],
    })

    stats_df = summary_df.describe()

    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        200
    )

    print("\n")

    print(stats_df.round(6))

    # ========================================================
    # Save
    # ========================================================
    summary_df.to_csv(
        f"{OUTPUT_DIR}/fold_results.csv",
        index=False
    )

    stats_df.to_csv(
        f"{OUTPUT_DIR}/summary_statistics.csv"
    )

    with open(
        f"{OUTPUT_DIR}/cv_results.json",
        "w"
    ) as f:

        json.dump(
            cv_results,
            f,
            indent=2
        )

    print("\nSaved Results")

    print(f"{OUTPUT_DIR}/fold_results.csv")

    print(f"{OUTPUT_DIR}/summary_statistics.csv")

    print(f"{OUTPUT_DIR}/cv_results.json")


if __name__ == "__main__":

    main()

