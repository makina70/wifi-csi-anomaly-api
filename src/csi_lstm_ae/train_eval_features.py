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

from csi_lstm_ae.data import (
    FEATURE_NAMES,
    fit_feature_normalizer,
    load_dataset,
    make_window_features,
    make_windows,
)
from csi_lstm_ae.model import FeatureAutoencoder
from csi_lstm_ae.train_eval import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a feature autoencoder on CSI window features.")
    parser.add_argument("--data", default="data/ex1_dataset.json")
    parser.add_argument("--out", default="outputs/ex1_feature_ae")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(values: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size, shuffle=shuffle)


def score_model(model: nn.Module, values: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in make_loader(values, batch_size, shuffle=False):
            batch = batch.to(device)
            pred = model(batch)
            scores.append(torch.mean((pred - batch) ** 2, dim=1).cpu().numpy())
    return np.concatenate(scores)


def plot_scores(centers: np.ndarray, labels: np.ndarray, scores: np.ndarray, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(centers[labels == 0], scores[labels == 0], s=14, label="normal", alpha=0.75)
    ax.scatter(centers[labels == 1], scores[labels == 1], s=14, label="abnormal", alpha=0.75)
    ax.axhline(threshold, color="black", linestyle="--", label="threshold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("feature reconstruction MSE")
    ax.set_title("Feature-AE reconstruction error over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_histogram(labels: np.ndarray, scores: np.ndarray, threshold: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores[labels == 0], bins=40, alpha=0.7, label="normal")
    ax.hist(scores[labels == 1], bins=40, alpha=0.7, label="abnormal")
    ax.axvline(threshold, color="black", linestyle="--", label="threshold")
    ax.set_xlabel("feature reconstruction MSE")
    ax.set_ylabel("window count")
    ax.set_title("Feature-AE score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data)
    windowed = make_windows(dataset, args.window_size, args.stride)
    features_raw = make_window_features(windowed.windows)

    normal_features = features_raw[windowed.labels == 0]
    split = int(len(normal_features) * 0.8)
    train_raw = normal_features[:split]
    val_raw = normal_features[split:]

    normalizer = fit_feature_normalizer(train_raw)
    train = normalizer.transform(train_raw).astype(np.float32)
    val = normalizer.transform(val_raw).astype(np.float32)
    all_features = normalizer.transform(features_raw).astype(np.float32)

    device = torch.device(args.device)
    model = FeatureAutoencoder(input_size=all_features.shape[1], latent_size=args.latent_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    losses: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for (batch,) in make_loader(train, args.batch_size, shuffle=True):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch)
            count += len(batch)

        train_loss = loss_sum / max(1, count)
        val_loss = float(score_model(model, val, args.batch_size, device).mean())
        losses.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    val_scores = score_model(model, val, args.batch_size, device)
    threshold = float(np.quantile(val_scores, args.threshold_quantile))
    scores = score_model(model, all_features, args.batch_size, device)
    predictions = (scores > threshold).astype(np.int64)
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
        "features": FEATURE_NAMES,
        "model": {
            "type": "Feature Autoencoder",
            "latent_size": args.latent_size,
            "epochs": args.epochs,
            "threshold_quantile": args.threshold_quantile,
        },
        "threshold": threshold,
        "metrics": metrics,
        "score_stats": {
            "normal_mean": float(scores[windowed.labels == 0].mean()),
            "normal_p99": float(np.quantile(scores[windowed.labels == 0], 0.99)),
            "abnormal_mean": float(scores[windowed.labels == 1].mean()),
            "abnormal_p99": float(np.quantile(scores[windowed.labels == 1], 0.99)),
        },
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "normalizer": {
                "mean": normalizer.mean.tolist(),
                "std": normalizer.std.tolist(),
            },
            "config": vars(args),
            "threshold": threshold,
        },
        output_dir / "model.pt",
    )
    np.savez(
        output_dir / "scores.npz",
        centers_sec=windowed.centers_sec,
        labels=windowed.labels,
        scores=scores,
        predictions=predictions,
        features_raw=features_raw,
    )
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "losses.json").write_text(json.dumps(losses, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_scores(windowed.centers_sec, windowed.labels, scores, threshold, output_dir / "scores_timeline.png")
    plot_histogram(windowed.labels, scores, threshold, output_dir / "score_histogram.png")

    print(json.dumps(metrics, indent=2))
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
