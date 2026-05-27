from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CsiDataset:
    metadata: dict
    time: np.ndarray
    signal: np.ndarray
    midpoint_index: int


@dataclass(frozen=True)
class WindowedCsi:
    windows: np.ndarray
    labels: np.ndarray
    starts: np.ndarray
    centers_sec: np.ndarray


def load_dataset(path: str | Path) -> CsiDataset:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)

    matrix = np.asarray(payload["data_matrix"], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("data_matrix must have at least two columns: time and signal")

    signal = matrix[:, 1].astype(np.float32)
    time = matrix[:, 0].astype(np.float32)
    midpoint_index = len(signal) // 2

    return CsiDataset(
        metadata=payload.get("metadata", {}),
        time=time,
        signal=signal,
        midpoint_index=midpoint_index,
    )


def make_windows(
    dataset: CsiDataset,
    window_size: int,
    stride: int,
    drop_boundary_crossing: bool = True,
) -> WindowedCsi:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    starts: list[int] = []
    centers: list[float] = []

    for start in range(0, len(dataset.signal) - window_size + 1, stride):
        end = start + window_size

        crosses_midpoint = start < dataset.midpoint_index < end
        if drop_boundary_crossing and crosses_midpoint:
            continue

        center_index = start + window_size // 2
        label = 0 if center_index < dataset.midpoint_index else 1

        windows.append(dataset.signal[start:end, None])
        labels.append(label)
        starts.append(start)
        centers.append(float(dataset.time[center_index]))

    return WindowedCsi(
        windows=np.asarray(windows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        centers_sec=np.asarray(centers, dtype=np.float32),
    )


@dataclass(frozen=True)
class Normalizer:
    mean: float
    std: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def fit_normalizer(windows: np.ndarray) -> Normalizer:
    mean = float(windows.mean())
    std = float(windows.std())
    if std == 0:
        std = 1.0
    return Normalizer(mean=mean, std=std)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "abs_mean",
    "max_abs",
    "p95_abs",
    "p99_abs",
    "range",
    "diff_std",
    "diff_rms",
    "diff_max_abs",
]

DIFF_FEATURE_NAMES = [
    "diff_mean",
    "diff_std",
    "diff_rms",
    "diff_abs_mean",
    "diff_max_abs",
    "diff_p95_abs",
    "diff_p99_abs",
    "diff_range",
    "diff_median_abs",
    "diff_mad",
]

ROBUST_FEATURE_NAMES = [
    "median",
    "iqr",
    "p95_abs",
    "p99_abs",
    "range",
    "diff_abs_mean",
    "diff_p95_abs",
    "diff_p99_abs",
    "diff_range",
    "diff_mad",
]


def resolve_feature_names(feature_set: str) -> list[str]:
    if feature_set == "base":
        return FEATURE_NAMES
    if feature_set == "diff":
        return DIFF_FEATURE_NAMES
    if feature_set == "robust":
        return ROBUST_FEATURE_NAMES
    raise ValueError(f"Unknown feature set: {feature_set}")


def make_window_features(windows: np.ndarray, feature_set: str = "base") -> np.ndarray:
    signal = windows[:, :, 0]
    diff = np.diff(signal, axis=1)

    abs_signal = np.abs(signal)
    abs_diff = np.abs(diff)
    signal_q75 = np.quantile(signal, 0.75, axis=1)
    signal_q25 = np.quantile(signal, 0.25, axis=1)
    diff_median = np.median(diff, axis=1, keepdims=True)

    feature_map = {
        "mean": np.mean(signal, axis=1),
        "std": np.std(signal, axis=1),
        "rms": np.sqrt(np.mean(signal**2, axis=1)),
        "abs_mean": np.mean(abs_signal, axis=1),
        "max_abs": np.max(abs_signal, axis=1),
        "p95_abs": np.quantile(abs_signal, 0.95, axis=1),
        "p99_abs": np.quantile(abs_signal, 0.99, axis=1),
        "range": np.max(signal, axis=1) - np.min(signal, axis=1),
        "median": np.median(signal, axis=1),
        "iqr": signal_q75 - signal_q25,
        "diff_mean": np.mean(diff, axis=1),
        "diff_std": np.std(diff, axis=1),
        "diff_rms": np.sqrt(np.mean(diff**2, axis=1)),
        "diff_abs_mean": np.mean(abs_diff, axis=1),
        "diff_max_abs": np.max(abs_diff, axis=1),
        "diff_p95_abs": np.quantile(abs_diff, 0.95, axis=1),
        "diff_p99_abs": np.quantile(abs_diff, 0.99, axis=1),
        "diff_range": np.max(diff, axis=1) - np.min(diff, axis=1),
        "diff_median_abs": np.median(abs_diff, axis=1),
        "diff_mad": np.median(np.abs(diff - diff_median), axis=1),
    }

    names = resolve_feature_names(feature_set)
    features = np.stack([feature_map[name] for name in names], axis=1)
    return features.astype(np.float32)


def fit_feature_normalizer(features: np.ndarray) -> FeatureNormalizer:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return FeatureNormalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))
