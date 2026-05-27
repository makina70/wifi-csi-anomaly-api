from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from csi_lstm_ae.data import load_dataset, make_window_features, make_windows
from csi_lstm_ae.model import FeatureAutoencoder
from csi_lstm_ae.train_eval import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Feature-AE on separate normal/abnormal files.")
    parser.add_argument("--model", default="models/feature_ae_w500/model.pt")
    parser.add_argument("--normal", required=True)
    parser.add_argument("--abnormal", required=True)
    parser.add_argument("--out", default="outputs/external_eval")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def score_file(path: str, checkpoint: dict, device: torch.device) -> tuple[np.ndarray, dict]:
    config = checkpoint.get("config", {})
    window_size = int(config.get("window_size", 500))
    stride = int(config.get("stride", 100))
    feature_set = str(config.get("feature_set", "base"))
    feature_names = checkpoint["feature_names"]

    dataset = load_dataset(path)
    # Treat each file as a single-class stream; overwrite labels outside this helper.
    dataset = type(dataset)(
        metadata=dataset.metadata,
        time=dataset.time,
        signal=dataset.signal,
        midpoint_index=len(dataset.signal),
    )
    windowed = make_windows(dataset, window_size=window_size, stride=stride, drop_boundary_crossing=False)
    features = make_window_features(windowed.windows, feature_set=feature_set)
    mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32)
    normalized = ((features - mean) / std).astype(np.float32)

    model = FeatureAutoencoder(input_size=len(feature_names), latent_size=int(config.get("latent_size", 3))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(normalized).to(device)
        reconstruction = model(tensor)
        scores = torch.mean((reconstruction - tensor) ** 2, dim=1).cpu().numpy()

    info = {
        "path": str(Path(path).resolve()),
        "metadata": dataset.metadata,
        "samples": int(len(dataset.signal)),
        "windows": int(len(scores)),
        "score_mean": float(scores.mean()),
        "score_p50": float(np.quantile(scores, 0.5)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "score_max": float(scores.max()),
    }
    return scores, info


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = torch.load(args.model, map_location=device)
    threshold = float(checkpoint["threshold"])

    normal_scores, normal_info = score_file(args.normal, checkpoint, device)
    abnormal_scores, abnormal_info = score_file(args.abnormal, checkpoint, device)

    labels = np.concatenate(
        [
            np.zeros_like(normal_scores, dtype=np.int64),
            np.ones_like(abnormal_scores, dtype=np.int64),
        ]
    )
    scores = np.concatenate([normal_scores, abnormal_scores])
    predictions = (scores > threshold).astype(np.int64)
    metrics = classification_metrics(labels, predictions)

    summary = {
        "model": str(Path(args.model).resolve()),
        "threshold": threshold,
        "feature_names": checkpoint["feature_names"],
        "config": checkpoint.get("config", {}),
        "normal": normal_info,
        "abnormal": abnormal_info,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(
        output_dir / "scores.npz",
        normal_scores=normal_scores,
        abnormal_scores=abnormal_scores,
        labels=labels,
        scores=scores,
        predictions=predictions,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
