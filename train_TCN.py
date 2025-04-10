"""
Trains a Temporal Convolutional Network (TCN) to forecast 26-week
violent-conflict dynamics (6 targets) from 52-weeks of history.

- Reads hyper-parameters from a YAML config.
- Uses mixed-precision (AMP) + cuDNN autotuner for speed.
- Saves metrics and flat per-week predictions to `output.directory.

Run: "python train_tcn.py --config config.yaml"
"""


from __future__ import annotations  # postpone evaluation of type hints

import argparse
import os
import time
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

torch.backends.cudnn.benchmark = True  # pick fastest conv kernels

# Dataset
class ConflictTCNDataset(Dataset):
    """Wrap (N, 52, 7) → (N, 26, 6) NumPy arrays as a PyTorch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:  # noqa: D401 – simple method
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# Split

def build_split(index_df: pd.DataFrame, seq_len: int, forecast_horizon: int):
    """
    Return train and test indices based on correct date.
    """
    index_df = index_df.copy()
    index_df["history_start"] = pd.to_datetime(index_df["history_start"])

    index_df["forecast_start"] = index_df["history_start"] + pd.Timedelta(weeks=seq_len)

    test_cutoff = index_df["forecast_start"].max()  # expected 2024‑08‑05

    train_mask = index_df["forecast_start"] < test_cutoff
    test_mask = ~train_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    assert test_idx.size > 0, "No sequences found for the prediction horizon."
    return train_idx, test_idx


# Main

def main(cfg: dict):
    # Load arrays, metadata
    X = np.load(cfg["data"]["X_path"], mmap_mode="r")  # (N, 52, 7)
    y = np.load(cfg["data"]["y_path"], mmap_mode="r")  # (N, 26, 6)
    index_df = pd.read_csv(cfg["data"]["sequence_index"])  # must have history_start

    assert len(X) == len(index_df), "sequence_index length mismatch with X/y"

    # Time-based split
    SEQ_LEN = 52
    F_HORIZ = cfg["model"]["forecast_horizon"]  # 26
    train_idx, test_idx = build_split(index_df, SEQ_LEN, F_HORIZ)

    ds = ConflictTCNDataset(X, y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin = device.type == "cuda"
    train_dl = DataLoader(Subset(ds, train_idx),
                          batch_size=cfg["training"]["batch_size"],
                          shuffle=True, pin_memory=pin, num_workers=os.cpu_count() // 2)
    test_dl = DataLoader(Subset(ds, test_idx),
                         batch_size=cfg["training"]["batch_size"],
                         shuffle=False, pin_memory=pin, num_workers=os.cpu_count() // 2)

    test_meta = index_df.iloc[test_idx].reset_index(drop=True)

    # Model
    from models.tcn import TemporalConvNet  # local import avoids circular deps

    model_core = TemporalConvNet(num_inputs=cfg["model"]["input_size"],
                                 num_channels=cfg["model"]["hidden_channels"],
                                 kernel_size=cfg["model"]["kernel_size"],
                                 dropout=cfg["model"]["dropout"])
    final_conv = nn.Conv1d(cfg["model"]["hidden_channels"][-1],
                           cfg["model"]["output_size"], kernel_size=1)

    model = nn.Sequential(model_core, final_conv).to(device)

    # Optim setup
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.3, patience=3, verbose=True)
    scaler = GradScaler(enabled=device.type == "cuda")
    criterion = nn.MSELoss()

    # Train loop
    epochs = cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}"):
            xb = xb.to(device).permute(0, 2, 1)  # (B, C, T)
            yb = yb.to(device)

            optimizer.zero_grad()
            with autocast(enabled=device.type == "cuda"):
                out = model(xb)                      # (B, 6, 52)
                out = out[:, :, -F_HORIZ:].permute(0, 2, 1)  # (B, 26, 6)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg = epoch_loss / len(train_dl)
        scheduler.step(avg)
        print(f"Epoch {epoch:02d} | loss={avg:.5f} | time={(time.time()-start)/60:.1f} min")

    # Eval
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in tqdm(test_dl, desc="Evaluating"):
            xb = xb.to(device).permute(0, 2, 1)
            with autocast(enabled=device.type == "cuda"):
                pb = model(xb)
                pb = pb[:, :, -F_HORIZ:].permute(0, 2, 1)
            preds.append(pb.cpu().numpy())
            trues.append(yb.numpy())

    y_pred = np.concatenate(preds, axis=0)  # (samples, 26, 6)
    y_true = np.concatenate(trues, axis=0)

    # Metrics
    from utils.metrics import compute_metrics

    metrics_df = compute_metrics(y_true, y_pred, target_cols=cfg["targets"], flatten=True)
    out_dir = Path(cfg["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "tcn_metrics.csv", index=False)
    print("Metrics saved →", out_dir / "tcn_metrics.csv")

    # Inference export
    samples, steps, tgts = y_pred.shape
    flat_idx = test_meta.loc[test_meta.index.repeat(steps)].copy()
    flat_idx["week_offset"] = np.tile(np.arange(steps), len(test_meta))

    results = pd.DataFrame({
        "true": y_true.reshape(-1),
        "predicted": y_pred.reshape(-1),
    })
    results = pd.concat([flat_idx.reset_index(drop=True), results], axis=1)
    results.to_csv(out_dir / "tcn_inference.csv", index=False)
    print("Flat predictions saved →", out_dir / "tcn_inference.csv")

    # -------------------- Model checkpoint -----------------------------------
    torch.save(model.state_dict(), out_dir / "tcn_state_dict.pt")
    print("Checkpoint saved →", out_dir / "tcn_state_dict.pt")


# Entry-point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)

