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
    fit_feature_normalizer,
    load_dataset,
    make_window_features,
    make_windows,
    resolve_feature_names,
)
from csi_lstm_ae.model import FeatureAutoencoder
from csi_lstm_ae.train_eval import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a feature autoencoder on CSI window features.")
    parser.add_argument("--data", default="data/ex1_dataset.json")
    parser.add_argument("--train-data", action="append", default=[])
    parser.add_argument("--eval-data", action="append", default=[])
    parser.add_argument("--eval-normal", action="append", default=[])
    parser.add_argument("--eval-abnormal", action="append", default=[])
    parser.add_argument("--out", default="outputs/ex1_feature_ae")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-size", type=int, default=3)
    parser.add_argument("--feature-set", choices=["base", "diff", "robust"], default="base")
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


def load_feature_blocks(
    paths: list[str],
    *,
    window_size: int,
    stride: int,
    feature_set: str,
    forced_label: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    center_blocks: list[np.ndarray] = []
    dataset_summaries: list[dict] = []

    for path in paths:
        dataset = load_dataset(path)
        if forced_label is not None:
            midpoint_index = len(dataset.signal) if forced_label == 0 else 0
            dataset = type(dataset)(
                metadata=dataset.metadata,
                time=dataset.time,
                signal=dataset.signal,
                midpoint_index=midpoint_index,
            )

        windowed = make_windows(
            dataset,
            window_size=window_size,
            stride=stride,
            drop_boundary_crossing=forced_label is None,
        )
        features = make_window_features(windowed.windows, feature_set=feature_set)
        labels = windowed.labels if forced_label is None else np.full(len(features), forced_label, dtype=np.int64)

        feature_blocks.append(features)
        label_blocks.append(labels)
        center_blocks.append(windowed.centers_sec)
        dataset_summaries.append(
            {
                "path": str(Path(path).resolve()),
                "metadata": dataset.metadata,
                "samples": int(len(dataset.signal)),
                "windows": int(len(windowed.windows)),
                "normal_windows": int(np.sum(labels == 0)),
                "abnormal_windows": int(np.sum(labels == 1)),
            }
        )

    if not feature_blocks:
        return (
            np.empty((0, len(resolve_feature_names(feature_set))), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            dataset_summaries,
        )

    return (
        np.concatenate(feature_blocks, axis=0).astype(np.float32),
        np.concatenate(label_blocks, axis=0).astype(np.int64),
        np.concatenate(center_blocks, axis=0).astype(np.float32),
        dataset_summaries,
    )


def build_experiment_data(args: argparse.Namespace) -> dict:
    separate_mode = bool(args.train_data or args.eval_data or args.eval_normal or args.eval_abnormal)
    if not separate_mode:
        dataset = load_dataset(args.data)
        windowed = make_windows(dataset, args.window_size, args.stride)
        features = make_window_features(windowed.windows, feature_set=args.feature_set)
        return {
            "mode": "legacy_single_dataset",
            "train_features_raw": features[windowed.labels == 0],
            "eval_features_raw": features,
            "eval_labels": windowed.labels,
            "eval_centers_sec": windowed.centers_sec,
            "summary": {
                "train_datasets": [
                    {
                        "path": str(Path(args.data).resolve()),
                        "metadata": dataset.metadata,
                        "samples": int(len(dataset.signal)),
                        "windows": int(len(windowed.windows)),
                        "normal_windows": int(np.sum(windowed.labels == 0)),
                        "abnormal_windows": int(np.sum(windowed.labels == 1)),
                    }
                ],
                "eval_datasets": [
                    {
                        "path": str(Path(args.data).resolve()),
                        "metadata": dataset.metadata,
                        "samples": int(len(dataset.signal)),
                        "windows": int(len(windowed.windows)),
                        "normal_windows": int(np.sum(windowed.labels == 0)),
                        "abnormal_windows": int(np.sum(windowed.labels == 1)),
                    }
                ],
            },
        }

    if not args.train_data:
        raise ValueError("--train-data is required when using separate train/eval inputs")

    train_features_raw, train_labels, _, train_summary = load_feature_blocks(
        args.train_data,
        window_size=args.window_size,
        stride=args.stride,
        feature_set=args.feature_set,
        forced_label=None,
    )
    train_normals = train_features_raw[train_labels == 0]
    if len(train_normals) == 0:
        raise ValueError("No normal windows found in --train-data")

    eval_feature_parts: list[np.ndarray] = []
    eval_label_parts: list[np.ndarray] = []
    eval_center_parts: list[np.ndarray] = []
    eval_summary: list[dict] = []

    if args.eval_data:
        features, labels, centers, summary = load_feature_blocks(
            args.eval_data,
            window_size=args.window_size,
            stride=args.stride,
            feature_set=args.feature_set,
            forced_label=None,
        )
        if len(features):
            eval_feature_parts.append(features)
            eval_label_parts.append(labels)
            eval_center_parts.append(centers)
        eval_summary.extend(summary)

    if args.eval_normal:
        features, labels, centers, summary = load_feature_blocks(
            args.eval_normal,
            window_size=args.window_size,
            stride=args.stride,
            feature_set=args.feature_set,
            forced_label=0,
        )
        if len(features):
            eval_feature_parts.append(features)
            eval_label_parts.append(labels)
            eval_center_parts.append(centers)
        eval_summary.extend(summary)

    if args.eval_abnormal:
        features, labels, centers, summary = load_feature_blocks(
            args.eval_abnormal,
            window_size=args.window_size,
            stride=args.stride,
            feature_set=args.feature_set,
            forced_label=1,
        )
        if len(features):
            eval_feature_parts.append(features)
            eval_label_parts.append(labels)
            eval_center_parts.append(centers)
        eval_summary.extend(summary)

    if not eval_feature_parts:
        raise ValueError("At least one of --eval-data, --eval-normal, or --eval-abnormal is required in separate mode")

    return {
        "mode": "separate_train_eval",
        "train_features_raw": train_normals,
        "eval_features_raw": np.concatenate(eval_feature_parts, axis=0).astype(np.float32),
        "eval_labels": np.concatenate(eval_label_parts, axis=0).astype(np.int64),
        "eval_centers_sec": np.concatenate(eval_center_parts, axis=0).astype(np.float32),
        "summary": {
            "train_datasets": train_summary,
            "eval_datasets": eval_summary,
        },
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    feature_names = resolve_feature_names(args.feature_set)
    experiment = build_experiment_data(args)

    normal_features = experiment["train_features_raw"]
    split = int(len(normal_features) * 0.8)
    if split <= 0 or split >= len(normal_features):
        raise ValueError("Not enough normal windows to create train/validation split")
    train_raw = normal_features[:split]
    val_raw = normal_features[split:]

    normalizer = fit_feature_normalizer(train_raw)
    train = normalizer.transform(train_raw).astype(np.float32)
    val = normalizer.transform(val_raw).astype(np.float32)
    eval_features = normalizer.transform(experiment["eval_features_raw"]).astype(np.float32)

    device = torch.device(args.device)
    model = FeatureAutoencoder(input_size=eval_features.shape[1], latent_size=args.latent_size).to(device)
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
    scores = score_model(model, eval_features, args.batch_size, device)
    predictions = (scores > threshold).astype(np.int64)
    metrics = classification_metrics(experiment["eval_labels"], predictions)

    summary = {
        "data_mode": experiment["mode"],
        "datasets": experiment["summary"],
        "windowing": {
            "window_size": args.window_size,
            "stride": args.stride,
            "train_normal_windows": int(len(normal_features)),
            "train_windows": int(len(train_raw)),
            "validation_windows": int(len(val_raw)),
            "evaluation_windows": int(len(experiment["eval_features_raw"])),
            "evaluation_normal_windows": int(np.sum(experiment["eval_labels"] == 0)),
            "evaluation_abnormal_windows": int(np.sum(experiment["eval_labels"] == 1)),
        },
        "features": feature_names,
        "model": {
            "type": "Feature Autoencoder",
            "latent_size": args.latent_size,
            "epochs": args.epochs,
            "threshold_quantile": args.threshold_quantile,
            "feature_set": args.feature_set,
        },
        "threshold": threshold,
        "metrics": metrics,
        "score_stats": {
            "normal_mean": float(scores[experiment["eval_labels"] == 0].mean()),
            "normal_p99": float(np.quantile(scores[experiment["eval_labels"] == 0], 0.99)),
            "abnormal_mean": float(scores[experiment["eval_labels"] == 1].mean()),
            "abnormal_p99": float(np.quantile(scores[experiment["eval_labels"] == 1], 0.99)),
        },
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": feature_names,
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
        centers_sec=experiment["eval_centers_sec"],
        labels=experiment["eval_labels"],
        scores=scores,
        predictions=predictions,
        features_raw=experiment["eval_features_raw"],
    )
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "losses.json").write_text(json.dumps(losses, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_scores(experiment["eval_centers_sec"], experiment["eval_labels"], scores, threshold, output_dir / "scores_timeline.png")
    plot_histogram(experiment["eval_labels"], scores, threshold, output_dir / "score_histogram.png")

    print(json.dumps(metrics, indent=2))
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
