# Wi-Fi CSI LSTM Autoencoder

This project trains and evaluates Autoencoder-based anomaly detectors on preprocessed CSI time-series data.

The included sample dataset is `data/ex1_dataset.json`. The first half is treated as normal data, and the second half is treated as abnormal data where a person was moving.

## Data

Expected JSON shape:

```json
{
  "metadata": {
    "sampling_rate_hz": 100.0,
    "matrix_columns": ["Time (Seconds)", "PC1_Phase_Variation"],
    "matrix_shape": [40659, 2]
  },
  "data_matrix": [
    [0.0, -140.72],
    [0.01, 140.50]
  ]
}
```

The model uses only the second column, `PC1_Phase_Variation`, as a 1D time series.

## Methods

### LSTM Autoencoder

1. Split the series into the first half as normal and the second half as abnormal.
2. Cut the series into sliding windows.
3. Train the LSTM Autoencoder only on normal windows.
4. Compute reconstruction error for all windows.
5. Set the anomaly threshold from normal validation reconstruction errors.
6. Classify windows above the threshold as abnormal.

Default windowing:

- Sampling rate: 100 Hz
- Window size: 200 samples, about 2 seconds
- Stride: 50 samples, about 0.5 seconds

### Feature Autoencoder

The raw LSTM-AE baseline can miss many abnormal windows because this dataset contains strong spike/range changes that are easier to detect as window-level statistics than as exact sequence reconstruction.

The feature AE uses each window's statistical features:

- mean
- standard deviation
- RMS
- absolute mean
- maximum absolute value
- 95th percentile absolute value
- 99th percentile absolute value
- range
- first-difference standard deviation
- first-difference RMS
- first-difference maximum absolute value

It still follows the same anomaly detection principle: train only on normal windows, then classify windows with high reconstruction error as abnormal.

## Setup

Use Python 3.11+.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If PyTorch is already installed globally, the virtual environment is optional.

## Run

From this directory:

```sh
PYTHONPATH=src python3 -m csi_lstm_ae.train_eval
```

For a quicker smoke test:

```sh
PYTHONPATH=src python3 -m csi_lstm_ae.train_eval --epochs 5
```

Recommended feature-AE run for the current sample:

```sh
PYTHONPATH=src python3 -m csi_lstm_ae.train_eval_features \
  --epochs 300 \
  --window-size 500 \
  --stride 100 \
  --threshold-quantile 0.99 \
  --out outputs/ex1_feature_ae_w500
```

## Outputs

The default output directory is `outputs/ex1_lstm_ae`.

Generated files:

- `model.pt`: trained PyTorch model, normalizer, and threshold
- `metrics.json`: dataset metadata, threshold, and classification metrics
- `losses.json`: train/validation loss per epoch
- `scores.npz`: reconstruction scores per window
- `loss.png`: training curve
- `scores_timeline.png`: reconstruction error over time
- `score_histogram.png`: normal/abnormal score distribution

## Current Result

On `ex1_dataset.json`, the raw LSTM-AE baseline worked as a pipeline but was not sensitive enough:

```text
LSTM-AE, window_size=100, stride=25
accuracy: 0.658
precision: 0.913
recall: 0.350
f1: 0.506
```

The feature AE improved the result substantially:

```text
Feature-AE, window_size=500, stride=100
accuracy: 0.965
precision: 0.995
recall: 0.934
f1: 0.964
```

For the exhibition demo, the feature AE is the better first candidate. It is still AE-based, but it uses robust window-level CSI features instead of trying to reconstruct the noisy raw sequence directly.

The trained model used by the API server is stored at:

```text
models/feature_ae_w500/model.pt
```

## Interpretation

This is an anomaly detection setup, not a supervised classifier. The model learns normal CSI behavior from the normal half only. Abnormal detection is based on reconstruction error.

If the normal and abnormal score distributions overlap, adjust:

- `--window-size`
- `--stride`
- `--hidden-size`
- `--latent-size`
- `--threshold-quantile`

The next step for the exhibition system is to put the trained scoring logic behind an API that returns:

```json
{
  "status": "abnormal",
  "anomalyScore": 0.91,
  "reconstructionError": 0.034,
  "threshold": 0.018
}
```

## ML/API Server

This project also contains a small FastAPI server for connecting the trained Feature-AE to the web app.

API endpoints:

- `POST /csi`: receive preprocessed CSI samples from the GMKtec side.
- `GET /latest`: return the latest normal/abnormal decision for the web UI.
- `GET /health`: return model and buffer status.

The server keeps the latest 500 samples in memory. Once the buffer is full, it computes the same 11 window features used during training, runs the Feature-AE, and updates the latest result.

### Start Locally

```sh
PYTHONPATH=src MODEL_PATH=models/feature_ae_w500/model.pt \
  uvicorn csi_lstm_ae.api_server:app --host 0.0.0.0 --port 8001
```

Health check:

```sh
curl http://localhost:8001/health
```

Latest result:

```sh
curl http://localhost:8001/latest
```

### Start With Docker

On the server Mac mini:

```sh
docker compose up --build
```

The ML API will listen on:

```text
http://<server-mac-mini-ip>:8001
```

The display Mac mini's UI container should be started with:

```sh
ML_API_BASE_URL=http://<server-mac-mini-ip>:8001 docker compose up --build
```

Then the web app should use:

```text
/api/ml/latest
```

### Input From GMKtec

The preprocessing side should send preprocessed `PC1_Phase_Variation` samples:

```http
POST /csi
Content-Type: application/json
```

```json
{
  "samplingRateHz": 100,
  "timestamp": "2026-05-27T16:30:00+09:00",
  "pc1PhaseVariation": [-140.72, 140.5, 1.03]
}
```

Recommended chunk size is 100 samples per request, which corresponds to about 1 second at 100 Hz.

### Output To Web App

`GET /latest` returns:

```json
{
  "status": "abnormal",
  "anomalyScore": 1.0,
  "reconstructionError": 0.84,
  "threshold": 0.5772173318266867,
  "timestamp": "2026-05-27T16:30:00+09:00",
  "samplesBuffered": 500,
  "windowSize": 500,
  "featureNames": ["mean", "std", "rms"]
}
```

Before 500 samples arrive, `status` is `warming_up`.

### Simulate CSI Streaming

With the API server running:

```sh
PYTHONPATH=src python3 -m csi_lstm_ae.simulate_stream \
  --data data/ex1_dataset.json \
  --url http://localhost:8001/csi \
  --chunk-size 100 \
  --sleep 1
```

Use `--sleep 0` to send the sample quickly.
