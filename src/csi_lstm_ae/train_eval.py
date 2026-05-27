from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from csi_lstm_ae.data import fit_normalizer, load_dataset, make_windows
from csi_lstm_ae.model import LSTMAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate an LSTM-AE on CSI data.")
    parser.add_argument("--data", default="data/ex1_dataset.json", help="Path to ex1_dataset.json")
    parser.add_argument("--out", default="outputs/ex1_lstm_ae", help="Output directory")
    parser.add_argument("--window-size", type=int, default=200, help="Window size in samples")
    parser.add_argument("--stride", type=int, default=50, help="Window stride in samples")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_normal_windows(windows: np.ndarray, labels: np.ndarray, train_ratio: float = 0.8):
    normal_windows = windows[labels == 0]
    split_index = int(len(normal_windows) * train_ratio)
    return normal_windows[:split_index], normal_windows[split_index:]


def make_loader(windows: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.from_numpy(windows)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)


def reconstruction_scores(
    model: nn.Module,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    loader = make_loader(windows, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            pred = model(batch)
            mse = torch.mean((pred - batch) ** 2, dim=(1, 2))
            scores.append(mse.cpu().numpy())
    return np.concatenate(scores)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))

    accuracy = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_losses(losses: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([row["epoch"] for row in losses], [row["train_loss"] for row in losses], label="train")
    ax.plot([row["epoch"] for row in losses], [row["val_loss"] for row in losses], label="normal validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("LSTM-AE training loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scores(
    centers_sec: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    normal = labels == 0
    abnormal = labels == 1
    ax.scatter(centers_sec[normal], scores[normal], s=12, label="normal", alpha=0.75)
    ax.scatter(centers_sec[abnormal], scores[abnormal], s=12, label="abnormal", alpha=0.75)
    ax.axhline(threshold, color="black", linestyle="--", label="threshold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("Reconstruction error over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_histogram(labels: np.ndarray, scores: np.ndarray, threshold: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[labels == 0], bins=40, alpha=0.7, label="normal")
    ax.hist(scores[labels == 1], bins=40, alpha=0.7, label="abnormal")
    ax.axvline(threshold, color="black", linestyle="--", label="threshold")
    ax.set_xlabel("reconstruction MSE")
    ax.set_ylabel("window count")
    ax.set_title("Score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data)
    windowed = make_windows(dataset, window_size=args.window_size, stride=args.stride)

    train_windows_raw, val_windows_raw = split_normal_windows(windowed.windows, windowed.labels)
    normalizer = fit_normalizer(train_windows_raw)

    train_windows = normalizer.transform(train_windows_raw).astype(np.float32)
    val_windows = normalizer.transform(val_windows_raw).astype(np.float32)
    all_windows = normalizer.transform(windowed.windows).astype(np.float32)

    device = torch.device(args.device)
    model = LSTMAutoencoder(
        input_size=1,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    train_loader = make_loader(train_windows, batch_size=args.batch_size, shuffle=True)

    losses: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(batch)
            train_count += len(batch)

        train_loss = train_loss_sum / max(1, train_count)
        val_scores = reconstruction_scores(model, val_windows, args.batch_size, device)
        val_loss = float(val_scores.mean())
        losses.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    val_scores = reconstruction_scores(model, val_windows, args.batch_size, device)
    threshold = float(np.quantile(val_scores, args.threshold_quantile))
    all_scores = reconstruction_scores(model, all_windows, args.batch_size, device)
    predictions = (all_scores > threshold).astype(np.int64)
    metrics = classification_metrics(windowed.labels, predictions)

    summary = {
        "dataset": {
            "path": str(Path(args.data).resolve()),
            "metadata": dataset.metadata,
            "samples": int(len(dataset.signal)),
            "normal_samples": int(dataset.midpoint_index),
            "abnormal_samples": int(len(dataset.signal) - dataset.midpoint_index),
        },
        "windowing": {
            "window_size": args.window_size,
            "stride": args.stride,
            "total_windows": int(len(windowed.windows)),
            "normal_windows": int(np.sum(windowed.labels == 0)),
            "abnormal_windows": int(np.sum(windowed.labels == 1)),
        },
        "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
        "model": {
            "type": "LSTM Autoencoder",
            "hidden_size": args.hidden_size,
            "latent_size": args.latent_size,
            "epochs": args.epochs,
            "threshold_quantile": args.threshold_quantile,
        },
        "threshold": threshold,
        "metrics": metrics,
        "score_stats": {
            "normal_mean": float(all_scores[windowed.labels == 0].mean()),
            "normal_p99": float(np.quantile(all_scores[windowed.labels == 0], 0.99)),
            "abnormal_mean": float(all_scores[windowed.labels == 1].mean()),
            "abnormal_p99": float(np.quantile(all_scores[windowed.labels == 1], 0.99)),
        },
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalizer": {"mean": normalizer.mean, "std": normalizer.std},
            "config": vars(args),
            "threshold": threshold,
        },
        output_dir / "model.pt",
    )

    np.savez(
        output_dir / "scores.npz",
        centers_sec=windowed.centers_sec,
        labels=windowed.labels,
        scores=all_scores,
        predictions=predictions,
    )

    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "losses.json").write_text(
        json.dumps(losses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_losses(losses, output_dir / "loss.png")
    plot_scores(windowed.centers_sec, windowed.labels, all_scores, threshold, output_dir / "scores_timeline.png")
    plot_histogram(windowed.labels, all_scores, threshold, output_dir / "score_histogram.png")

    print(json.dumps(summary["metrics"], indent=2))
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
