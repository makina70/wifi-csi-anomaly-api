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


class HealthResult(BaseModel):
    ok: bool
    modelPath: str
    samplesBuffered: int
    windowSize: int
    threshold: float


class CsiAnomalyService:
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self.model_path = Path(model_path)
        checkpoint = torch.load(self.model_path, map_location=device)
        config = checkpoint.get("config", {})

        self.window_size = int(config.get("window_size", 500))
        self.threshold = float(checkpoint["threshold"])
        self.feature_names = list(checkpoint.get("feature_names", FEATURE_NAMES))
        self.feature_set = str(config.get("feature_set", "base"))
        self.scoring = checkpoint.get("scoring", {"mode": "autoencoder"})
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
        self.lock = Lock()
        self.latest = LatestResult(
            status="warming_up",
            anomalyScore=0.0,
            reconstructionError=None,
            threshold=self.threshold,
            timestamp=utc_now(),
            samplesBuffered=0,
            windowSize=self.window_size,
            featureNames=self.feature_names,
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
                    threshold=self.threshold,
                    timestamp=timestamp or utc_now(),
                    samplesBuffered=len(self.buffer),
                    windowSize=self.window_size,
                    featureNames=self.feature_names,
                )
                return self.latest

            window = np.asarray(self.buffer, dtype=np.float32)[None, :, None]
            features = make_window_features(window, feature_set=self.feature_set)

            if self.scoring.get("mode") == "feature_threshold":
                error = self.score_feature_threshold(features[0])
            else:
                normalized = ((features - self.mean) / self.std).astype(np.float32)
                with torch.no_grad():
                    tensor = torch.from_numpy(normalized).to(self.device)
                    reconstruction = self.model(tensor)
                    error = float(torch.mean((reconstruction - tensor) ** 2).cpu().item())

            status: Literal["normal", "abnormal"] = "abnormal" if error > self.threshold else "normal"
            anomaly_score = min(1.0, max(0.0, error / self.threshold)) if self.threshold > 0 else 0.0
            self.latest = LatestResult(
                status=status,
                anomalyScore=round(anomaly_score, 4),
                reconstructionError=error,
                threshold=self.threshold,
                timestamp=timestamp or utc_now(),
                samplesBuffered=len(self.buffer),
                windowSize=self.window_size,
                featureNames=self.feature_names,
            )
            return self.latest

    def score_feature_threshold(self, features: np.ndarray) -> float:
        thresholds = self.scoring.get("thresholds", {})
        if not thresholds:
            return 0.0

        feature_index = {name: index for index, name in enumerate(self.feature_names)}
        ratios: list[float] = []
        for name, threshold in thresholds.items():
            if name not in feature_index:
                continue
            threshold_value = float(threshold)
            if threshold_value <= 0:
                continue
            ratios.append(float(features[feature_index[name]] / threshold_value))

        return max(ratios) if ratios else 0.0

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
                threshold=self.threshold,
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
