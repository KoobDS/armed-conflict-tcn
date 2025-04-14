"""
Trains a Temporal Convolutional Network (TCN) to forecast 26-week
violent-conflict dynamics (6 targets) from 52-weeks of history.

- Reads hyper-parameters from a YAML config.
- Uses mixed-precision (AMP) + cuDNN autotuner for speed.
- Saves metrics and flat per-week predictions to `output.directory.

Run: "python train_tcn.py --config config.yaml"
"""


from __future__ import annotations
import argparse, math, json, time
from pathlib import Path

import yaml, numpy as np, pandas as pd, torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


### Dataset

class ConflictTCNDataset(Dataset):
    """52-week history → 26-week targets (memory-mapped, z-scored)."""
    def __init__(self, X, y, mean, std):
        self.X, self.y = X, y
        self.mean = mean.astype(np.float32)
        self.std  = np.clip(std, 1e-6, None).astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        return (torch.from_numpy(np.nan_to_num(x, copy=False)).float(),
                torch.from_numpy(self.y[i]).float())


### Helpers

def time_split(df: pd.DataFrame, seq_len=52):
    df = df.copy()
    df["history_start"]  = pd.to_datetime(df["history_start"])
    df["forecast_start"] = df["history_start"] + pd.Timedelta(weeks=seq_len)
    cut = df["forecast_start"].max()
    tr  = np.where(df["forecast_start"] <  cut)[0]
    te  = np.where(df["forecast_start"] == cut)[0]
    return tr, te

def streaming_mean_std(arr, rows, bs=8_000):
    s = s2 = 0.0; n = 0
    for i in range(0, len(rows), bs):
        b = arr[rows[i:i+bs]].reshape(-1, arr.shape[2])
        s  += b.sum(0);  s2 += (b**2).sum(0);  n += len(b)
    m = s / n
    v = np.maximum(s2 / n - m**2, 1e-12)
    return m, np.sqrt(v)

def build_loaders(cfg, X, y, idx):
    tr_idx, te_idx = time_split(idx)
    # 90-10 split for validation
    val_cut = int(len(tr_idx) * (1 - cfg["training"]["val_split"]))
    val_idx = tr_idx[val_cut:]; tr_idx = tr_idx[:val_cut]

    mean, std = streaming_mean_std(X, tr_idx)
    ds = ConflictTCNDataset(X, y, mean, std)

    def loader(idxs, shuffle):
        return DataLoader(Subset(ds, idxs),
                          batch_size=cfg["training"]["batch_size"],
                          shuffle=shuffle, num_workers=2,
                          pin_memory=True, persistent_workers=True)
    return loader(tr_idx, True), loader(val_idx, False), loader(te_idx, False), te_idx, mean, std

def build_model(cfg, device):
    from models.tcn import TemporalConvNet
    core = TemporalConvNet(cfg["model"]["input_size"],
                           cfg["model"]["hidden_channels"],
                           kernel_size=cfg["model"]["kernel_size"],
                           dropout=cfg["model"]["dropout"])
    head = nn.Conv1d(cfg["model"]["hidden_channels"][-1],
                     cfg["model"]["output_size"], 1)
    return nn.Sequential(core, head).to(device)

def warmup_cosine_lr(epoch:int,
                     total_epochs:int,
                     warmup_epochs:int,
                     base_lr:float|str,
                     min_lr:float = 1e-6) -> float:
    base_lr = float(base_lr)
    if epoch <= warmup_epochs:
        return min_lr + (base_lr - min_lr) * (epoch / warmup_epochs)

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    cosine   = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine

### Main (Train/Eval)

def run(cfg, eval_only=False):
    # ---------- data ----------
    X   = np.load(cfg["data"]["X_path"], mmap_mode="r")
    y   = np.load(cfg["data"]["y_path"], mmap_mode="r")
    idx = pd.read_csv(cfg["data"]["sequence_index"])
    dl_tr, dl_val, dl_te, te_idx, mean, std = build_loaders(cfg, X, y, idx)

    # ---------- model ----------
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, dev)
    FH    = cfg["model"]["forecast_horizon"]
    out   = Path(cfg["output"]["directory"]); out.mkdir(parents=True, exist_ok=True)

    if not eval_only:
        opt     = optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))
        scaler  = GradScaler(enabled=dev.type == "cuda")
        crit    = nn.MSELoss()
        E       = cfg["training"]["epochs"]
        W       = cfg["training"]["warmup_epochs"]

        losses, val_losses = [], []
        print("Training TCN...")
        for epoch in range(1, E + 1):
            # ---- set LR ----
            lr = warmup_cosine_lr(epoch, E, W, cfg["training"]["lr"])
            for pg in opt.param_groups: pg["lr"] = lr

            # ---- train ----
            model.train(); tot=0; t0=time.time()
            for xb,yb in tqdm(dl_tr, desc=f"Epoch {epoch:02d}", leave=False):
                xb,yb = xb.to(dev).permute(0,2,1), yb.to(dev)
                opt.zero_grad(set_to_none=True)
                with autocast(device_type=dev.type, enabled=dev.type=="cuda"):
                    pred = model(xb)[:,:, -FH:].permute(0,2,1)
                    loss = crit(pred, yb)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); tot += loss.item()
            train_loss = tot/len(dl_tr); losses.append(train_loss)

            # ---- val ----
            model.eval(); vtot=0
            with torch.no_grad():
                for xb,yb in dl_val:
                    xb,yb = xb.to(dev).permute(0,2,1), yb.to(dev)
                    with autocast(device_type=dev.type, enabled=dev.type=="cuda"):
                        vloss = crit(model(xb)[:,:, -FH:].permute(0,2,1), yb)
                    vtot += vloss.item()
            val_loss = vtot/len(dl_val); val_losses.append(val_loss)

            print(f"Epoch {epoch:02d} | Train {train_loss:.3f} | Val {val_loss:.3f} | "
                  f"LR {lr:.2e} | {(time.time()-t0)/60:.1f} min")

            # save checkpoint each epoch (optional)
            torch.save(model.state_dict(), out/'best.pt')

        json.dump({"train": losses, "val": val_losses},
                  open(out/'train_losses.json','w'), indent=2)

    else:
        print("Eval-only mode — loading best.pt")
        model.load_state_dict(torch.load(out/'best.pt'))

    # ---------- evaluation ----------
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb,yb in tqdm(dl_te, desc="Evaluating"):
            xb = xb.to(dev).permute(0,2,1)
            with autocast(device_type=dev.type, enabled=dev.type=="cuda"):
                p = model(xb)[:,:, -FH:].permute(0,2,1)
            preds.append(p.cpu().numpy()); trues.append(yb.numpy())
    y_pred, y_true = map(lambda a: np.nan_to_num(np.concatenate(a)), (preds, trues))

    from metrics import compute_metrics
    m_df = compute_metrics(y_true, y_pred, target_cols=cfg["targets"])
    m_df.to_csv(out/'tcn_metrics.csv', index=False)

    # ----- Inference output (fixed)
    samples, steps, n_tgt = y_pred.shape

    # Long format predictions
    df_pred = (pd.DataFrame(y_pred.reshape(samples * steps, n_tgt),
                            columns=cfg["targets"])
            .stack().reset_index(name="predicted"))  # [sample_step, event, predicted]

    df_true = (pd.DataFrame(y_true.reshape(samples * steps, n_tgt),
                            columns=cfg["targets"])
            .stack().reset_index(name="true"))  # [sample_step, event, true]

    # Merge
    df_pred.columns = ["flat_index", "event", "predicted"]
    df_true.columns = ["flat_index", "event", "true"]
    long_df = df_true.join(df_pred["predicted"])

    # Add prediction_week and true sample index
    long_df["prediction_week"] = long_df["flat_index"] % steps + 1
    long_df["sample_idx"] = long_df["flat_index"] // steps
    long_df.drop(columns="flat_index", inplace=True)

    # Metadata — repeat each sample × steps × targets
    meta = idx.iloc[te_idx][["country", "admin1"]].reset_index(drop=True)
    meta_full = meta.loc[meta.index.repeat(steps * n_tgt)].reset_index(drop=True)

    # Combine all
    long_df = pd.concat([meta_full, long_df[["prediction_week", "event", "true", "predicted"]]], axis=1)

    # Save
    long_df.to_csv(out / "tcn_inference.csv", index=False)
    print("Final cleaned inference saved →", out / "tcn_inference.csv")

# ──────────────────────────── CLI ───────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, eval_only=args.eval_only)