# Wi-Fi CSI Anomaly API

This repository contains the Mac mini side of the Wi-Fi CSI anomaly detection system.

The current runtime does not make the final decision from autoencoder reconstruction error. It receives `motionScore` samples from the GMKtec CSI agent, calibrates an empty-room baseline, and detects sustained motion using adaptive feature thresholds.

## Role

```text
GMKtec:
  PicoScenes -> .csi -> motionScore -> POST /csi

Mac mini:
  POST /csi -> baseline calibration -> diff_mad scoring -> normal/abnormal
```

## Input

The GMKtec agent posts to:

```text
POST /csi
```

Payload:

```json
{
  "samplingRateHz": 100.0,
  "pc1PhaseVariation": [1.23, 1.41, 1.36],
  "timestamp": "2026-05-30T06:00:00+00:00"
}
```

For compatibility, the field is still named `pc1PhaseVariation`, but the values are `motionScore` samples generated on the GMKtec side.

## Runtime Detection

The API keeps the latest `500` samples in memory.

After startup, the first `100` full windows are used for empty-room calibration. During this period, the room should be empty.

For each 500-sample window, the API computes robust features:

```text
median
iqr
p95_abs
p99_abs
range
diff_abs_mean
diff_p95_abs
diff_p99_abs
diff_range
diff_mad
```

The current final decision uses `diff_mad` as the motion evidence feature.

The empty-room baseline creates adaptive thresholds from the first 100 windows:

```text
upper = 90th percentile * 1.05
lower = 20th percentile * 0.90
```

For `diff_mad`, the score is based on the current value relative to its calibrated upper threshold.

Current decision rule:

```text
anomalyScore >= 0.40
```

is treated as motion evidence. The API returns `abnormal` only when this happens in at least `3` of the latest `5` windows.

This sustained-window rule is used to avoid false positives from one-off noise spikes.

## Current Docker Settings

The active runtime settings are in `docker-compose.yml`:

```yaml
ANOMALY_SCORE_THRESHOLD: "1.00"
STATUS_SCORE_THRESHOLD: "0.40"
ADAPTIVE_FEATURE_CALIBRATION: "true"
CALIBRATION_WINDOWS: "100"
CALIBRATION_LOWER_QUANTILE: "0.20"
CALIBRATION_LOWER_MULTIPLIER: "0.90"
CALIBRATION_QUANTILE: "0.90"
CALIBRATION_MULTIPLIER: "1.05"
LOWER_DEVIATION_FEATURES: "p95_abs,p99_abs,range"
MOTION_EVIDENCE_FEATURES: "diff_mad"
MOTION_EVIDENCE_THRESHOLD: "0.75"
MOTION_EVIDENCE_WINDOWS: "5"
MOTION_EVIDENCE_REQUIRED: "3"
```

`STATUS_SCORE_THRESHOLD` controls the normal/abnormal decision.

`ANOMALY_SCORE_THRESHOLD` is kept for score scaling compatibility and should remain `1.00`.

## Output

`GET /latest` returns the latest result:

```json
{
  "status": "normal",
  "anomalyScore": 0.08,
  "reconstructionError": 0.08,
  "threshold": 0.4,
  "timestamp": "2026-05-30T06:00:00+00:00",
  "samplesBuffered": 500,
  "windowSize": 500,
  "featureNames": ["median", "iqr", "p95_abs", "p99_abs", "range", "diff_abs_mean", "diff_p95_abs", "diff_p99_abs", "diff_range", "diff_mad"],
  "featureValues": {
    "diff_mad": 0.012
  },
  "featureRatios": {
    "diff_mad": 0.06
  },
  "adaptiveThresholds": {
    "diff_mad": 0.2
  },
  "calibrationWindows": 100,
  "calibrationRequiredWindows": 100
}
```

The field `reconstructionError` remains in the API response for compatibility with the existing UI. In the current runtime mode, it is the same motion evidence score used for anomaly detection, not an autoencoder reconstruction error.

## API Endpoints

```text
POST /csi
GET  /latest
GET  /health
```

## Run On The Mac Mini

```sh
docker compose up -d --build
```

The API listens on:

```text
http://localhost:8001
```

Health check:

```sh
curl http://localhost:8001/health
```

Latest result:

```sh
curl http://localhost:8001/latest
```

## Update From GitHub

On the Mac mini:

```sh
git pull
docker compose down
docker compose up -d --build
```

## Notes

Legacy training and evaluation scripts remain in `src/csi_lstm_ae/` for reference, but the current real-time deployment path is:

```text
GMKtec motionScore -> adaptive diff_mad scoring -> sustained normal/abnormal decision
```
