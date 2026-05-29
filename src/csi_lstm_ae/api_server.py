from __future__ import annotations

import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from csi_lstm_ae.data import FEATURE_NAMES, make_window_features
from csi_lstm_ae.model import FeatureAutoencoder


DEFAULT_MODEL_PATH = "models/feature_ae_w500/model.pt"


class CsiPayload(BaseModel):
    samplingRateHz: float = Field(default=100.0, gt=0)
    pc1PhaseVariation: list[float] = Field(min_length=1)
    timestamp: str | None = None


class LatestResult(BaseModel):
    status: Literal["normal", "abnormal", "warming_up"]
    anomalyScore: float
    reconstructionError: float | None
    threshold: float
    timestamp: str
    samplesBuffered: int
    windowSize: int
    featureNames: list[str]
    featureValues: dict[str, float] | None = None
    featureRatios: dict[str, float] | None = None
    adaptiveThresholds: dict[str, float] | None = None
    adaptiveLowerThresholds: dict[str, float] | None = None
    calibrationWindows: int = 0
    calibrationRequiredWindows: int = 0


class HealthResult(BaseModel):
    ok: bool
    modelPath: str
    samplesBuffered: int
    windowSize: int
    threshold: float
    scoringMode: str
    adaptiveThresholds: dict[str, float] | None = None
    adaptiveLowerThresholds: dict[str, float] | None = None
    calibrationWindows: int = 0
    calibrationRequiredWindows: int = 0


class CsiAnomalyService:
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        checkpoint = torch.load(self.model_path, map_location=device)
        config = checkpoint.get("config", {})

        self.window_size = int(config.get("window_size", 500))
        self.threshold = float(os.getenv("ANOMALY_SCORE_THRESHOLD", checkpoint["threshold"]))
        self.status_threshold = float(os.getenv("STATUS_SCORE_THRESHOLD", self.threshold))
        self.feature_names = list(checkpoint.get("feature_names", FEATURE_NAMES))
        self.feature_set = str(config.get("feature_set", "base"))
        self.scoring = checkpoint.get("scoring", {"mode": "autoencoder"})
        self.adaptive_calibration_enabled = (
            self.scoring.get("mode") == "feature_threshold"
            and os.getenv("ADAPTIVE_FEATURE_CALIBRATION", "true").lower() in {"1", "true", "yes", "on"}
        )
        self.calibration_required_windows = int(os.getenv("CALIBRATION_WINDOWS", "20"))
        self.calibration_lower_quantile = float(os.getenv("CALIBRATION_LOWER_QUANTILE", "0.20"))
        self.calibration_quantile = float(os.getenv("CALIBRATION_QUANTILE", "0.80"))
        self.calibration_lower_multiplier = float(os.getenv("CALIBRATION_LOWER_MULTIPLIER", "0.80"))
        self.calibration_multiplier = float(os.getenv("CALIBRATION_MULTIPLIER", "1.00"))
        self.lower_deviation_features = parse_csv_env("LOWER_DEVIATION_FEATURES")
        self.motion_evidence_features = parse_csv_env("MOTION_EVIDENCE_FEATURES")
        self.motion_evidence_threshold = float(os.getenv("MOTION_EVIDENCE_THRESHOLD", "1.00"))
        self.motion_evidence_windows = int(os.getenv("MOTION_EVIDENCE_WINDOWS", "1"))
        self.motion_evidence_required = int(os.getenv("MOTION_EVIDENCE_REQUIRED", "1"))
        self.calibration_values: list[dict[str, float]] = []
        self.adaptive_lower_thresholds: dict[str, float] | None = None
        self.adaptive_thresholds: dict[str, float] | None = None
        self.mean = np.asarray(checkpoint["normalizer"]["mean"], dtype=np.float32)
        self.std = np.asarray(checkpoint["normalizer"]["std"], dtype=np.float32)
        self.device = torch.device(device)

        self.model = FeatureAutoencoder(
            input_size=len(self.feature_names),
            latent_size=int(config.get("latent_size", 3)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.buffer: deque[float] = deque(maxlen=self.window_size)
        self.motion_evidence: deque[bool] = deque(maxlen=max(1, self.motion_evidence_windows))
        self.lock = Lock()
        self.latest = LatestResult(
            status="warming_up",
            anomalyScore=0.0,
            reconstructionError=None,
            threshold=self.status_threshold,
            timestamp=utc_now(),
            samplesBuffered=0,
            windowSize=self.window_size,
            featureNames=self.feature_names,
            calibrationRequiredWindows=self.calibration_required_windows if self.adaptive_calibration_enabled else 0,
        )

    def add_samples(self, samples: list[float], timestamp: str | None = None) -> LatestResult:
        clean_samples = [float(value) for value in samples if np.isfinite(value)]
        with self.lock:
            self.buffer.extend(clean_samples)
            if len(self.buffer) < self.window_size:
                self.latest = LatestResult(
                    status="warming_up",
                    anomalyScore=0.0,
                    reconstructionError=None,
                    threshold=self.status_threshold,
                    timestamp=timestamp or utc_now(),
                    samplesBuffered=len(self.buffer),
                    windowSize=self.window_size,
                    featureNames=self.feature_names,
                    calibrationWindows=len(self.calibration_values),
                    calibrationRequiredWindows=self.calibration_required_windows if self.adaptive_calibration_enabled else 0,
                )
                return self.latest

            window = np.asarray(self.buffer, dtype=np.float32)[None, :, None]
            features = make_window_features(window, feature_set=self.feature_set)
            feature_values = self.selected_feature_values(features[0])

            if self.adaptive_calibration_enabled and self.adaptive_thresholds is None:
                self.calibration_values.append(feature_values)
                if len(self.calibration_values) < self.calibration_required_windows:
                    self.latest = LatestResult(
                        status="warming_up",
                        anomalyScore=0.0,
                        reconstructionError=None,
                        threshold=self.status_threshold,
                        timestamp=timestamp or utc_now(),
                        samplesBuffered=len(self.buffer),
                        windowSize=self.window_size,
                        featureNames=self.feature_names,
                        featureValues=feature_values,
                        adaptiveThresholds=self.adaptive_thresholds,
                        adaptiveLowerThresholds=self.adaptive_lower_thresholds,
                        calibrationWindows=len(self.calibration_values),
                        calibrationRequiredWindows=self.calibration_required_windows,
                    )
                    return self.latest
                self.adaptive_lower_thresholds, self.adaptive_thresholds = self.compute_adaptive_thresholds()

            if self.scoring.get("mode") == "feature_threshold":
                error, feature_ratios = self.score_feature_threshold(features[0])
                if self.motion_evidence_features:
                    error = self.score_motion_evidence(feature_ratios)
            else:
                normalized = ((features - self.mean) / self.std).astype(np.float32)
                with torch.no_grad():
                    tensor = torch.from_numpy(normalized).to(self.device)
                    reconstruction = self.model(tensor)
                    error = float(torch.mean((reconstruction - tensor) ** 2).cpu().item())
                feature_ratios = None

            if self.motion_evidence_features:
                is_evidence = error >= self.status_threshold
                self.motion_evidence.append(is_evidence)
                required = min(self.motion_evidence_required, self.motion_evidence.maxlen or 1)
                status: Literal["normal", "abnormal"] = (
                    "abnormal" if sum(self.motion_evidence) >= required else "normal"
                )
            else:
                status = "abnormal" if error > self.status_threshold else "normal"
            anomaly_score = min(1.0, max(0.0, error / self.threshold)) if self.threshold > 0 else 0.0
            self.latest = LatestResult(
                status=status,
                anomalyScore=round(anomaly_score, 4),
                reconstructionError=error,
                threshold=self.status_threshold,
                timestamp=timestamp or utc_now(),
                samplesBuffered=len(self.buffer),
                windowSize=self.window_size,
                featureNames=self.feature_names,
                featureValues=feature_values,
                featureRatios=feature_ratios,
                adaptiveThresholds=self.adaptive_thresholds,
                adaptiveLowerThresholds=self.adaptive_lower_thresholds,
                calibrationWindows=len(self.calibration_values),
                calibrationRequiredWindows=self.calibration_required_windows if self.adaptive_calibration_enabled else 0,
            )
            return self.latest

    def selected_feature_values(self, features: np.ndarray) -> dict[str, float]:
        feature_index = {name: index for index, name in enumerate(self.feature_names)}
        selected_names = self.scoring.get("selected_features") or list(self.scoring.get("thresholds", {}))
        values: dict[str, float] = {}
        for name in selected_names:
            if name in feature_index:
                values[name] = float(features[feature_index[name]])
        return values

    def compute_adaptive_thresholds(self) -> tuple[dict[str, float], dict[str, float]]:
        lower_thresholds: dict[str, float] = {}
        upper_thresholds: dict[str, float] = {}
        selected_names = self.scoring.get("selected_features") or list(self.scoring.get("thresholds", {}))
        for name in selected_names:
            values = [entry[name] for entry in self.calibration_values if name in entry]
            if not values:
                continue
            values_array = np.asarray(values, dtype=np.float32)
            lower = float(np.quantile(values_array, self.calibration_lower_quantile))
            upper = float(np.quantile(values_array, self.calibration_quantile))
            lower_thresholds[name] = max(lower * self.calibration_lower_multiplier, 1e-6)
            upper_thresholds[name] = max(upper * self.calibration_multiplier, 1e-6)
        return lower_thresholds, upper_thresholds

    def score_feature_threshold(self, features: np.ndarray) -> tuple[float, dict[str, float]]:
        thresholds = self.adaptive_thresholds if self.adaptive_thresholds is not None else self.scoring.get("thresholds", {})
        if not thresholds:
            return 0.0, {}

        feature_index = {name: index for index, name in enumerate(self.feature_names)}
        ratios: list[float] = []
        feature_ratios: dict[str, float] = {}
        for name, threshold in thresholds.items():
            if name not in feature_index:
                continue
            threshold_value = float(threshold)
            if threshold_value <= 0:
                continue
            value = float(features[feature_index[name]])
            ratio = value / threshold_value
            if self.adaptive_lower_thresholds is not None and name in self.adaptive_lower_thresholds and value > 0:
                if name in self.lower_deviation_features:
                    ratio = max(ratio, float(self.adaptive_lower_thresholds[name]) / value)
            ratios.append(ratio)
            feature_ratios[name] = ratio

        return (max(ratios) if ratios else 0.0), feature_ratios

    def score_motion_evidence(self, feature_ratios: dict[str, float]) -> float:
        if self.motion_evidence_threshold <= 0:
            return 0.0
        evidence_ratios = [
            float(feature_ratios[name])
            for name in self.motion_evidence_features
            if name in feature_ratios
        ]
        if not evidence_ratios:
            return 0.0
        return max(evidence_ratios) / self.motion_evidence_threshold

    def get_latest(self) -> LatestResult:
        with self.lock:
            return self.latest

    def health(self) -> HealthResult:
        with self.lock:
            return HealthResult(
                ok=True,
                modelPath=str(self.model_path),
                samplesBuffered=len(self.buffer),
                windowSize=self.window_size,
                threshold=self.status_threshold,
                scoringMode=str(self.scoring.get("mode", "autoencoder")),
                adaptiveThresholds=self.adaptive_thresholds,
                adaptiveLowerThresholds=self.adaptive_lower_thresholds,
                calibrationWindows=len(self.calibration_values),
                calibrationRequiredWindows=self.calibration_required_windows if self.adaptive_calibration_enabled else 0,
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_env(name: str) -> set[str]:
    raw_value = os.getenv(name, "")
    return {value.strip() for value in raw_value.split(",") if value.strip()}


def create_app() -> FastAPI:
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    device = os.getenv("MODEL_DEVICE", "cpu")
    service = CsiAnomalyService(model_path=model_path, device=device)

    app = FastAPI(title="Wi-Fi CSI ML API", version="0.1.0")

    @app.post("/csi", response_model=LatestResult)
    def post_csi(payload: CsiPayload) -> LatestResult:
        return service.add_samples(payload.pc1PhaseVariation, timestamp=payload.timestamp)

    @app.get("/latest", response_model=LatestResult)
    def get_latest() -> LatestResult:
        return service.get_latest()

    @app.get("/health", response_model=HealthResult)
    def get_health() -> HealthResult:
        return service.health()

    return app


app = create_app()
