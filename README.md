# Wi-Fi CSI Anomaly API

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

For better robustness against baseline shifts in normal CSI, the current API model uses the `robust` feature set:

- median
- IQR
- 95th percentile absolute value
- 99th percentile absolute value
- range
- first-difference absolute mean
- first-difference 95th percentile absolute value
- first-difference 99th percentile absolute value
- first-difference range
- first-difference MAD

This keeps the AE approach but makes the input less dependent on the raw signal baseline. The first-difference features emphasize temporal change, which is closer to the actual movement cue.

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
  --feature-set robust \
  --epochs 300 \
  --window-size 500 \
  --stride 100 \
  --threshold-quantile 0.99 \
  --out outputs/ex1_feature_ae_robust_w500
```

If you want to separate training data from evaluation data, use `--train-data` and one or more evaluation inputs.
`--train-data` keeps only normal windows for training. Evaluation can use:

- `--eval-data`: mixed files that already contain normal/abnormal labels through `metadata.normal_samples`
- `--eval-normal`: files treated as fully normal
- `--eval-abnormal`: files treated as fully abnormal

Example:

```sh
PYTHONPATH=src python3 -m csi_lstm_ae.train_eval_features \
  --train-data data/ex1_dataset.json \
  --eval-normal /Users/uchimakikohki/Downloads/seijou1.json \
  --eval-abnormal /Users/uchimakikohki/Downloads/ijou1.json \
  --feature-set robust \
  --epochs 300 \
  --window-size 500 \
  --stride 100 \
  --threshold-quantile 0.90 \
  --out outputs/exhibition_split_eval
```

In this mode, training and threshold calibration come only from the training-side normal windows. Evaluation metrics are computed only on the explicitly supplied evaluation datasets.

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

The first feature AE improved the result substantially:

```text
Feature-AE, base features, window_size=500, stride=100
accuracy: 0.965
precision: 0.995
recall: 0.934
f1: 0.964
```

The robust/difference-focused Feature-AE improved it further:

```text
Feature-AE, robust features, window_size=500, stride=100
accuracy: 0.995
precision: 0.995
recall: 0.995
f1: 0.995
```

That score came from a single normal/abnormal pair. For deployment, the default API model now uses a held-out split built from the exhibition data:

- train normal:
  - first `80%` of the normal half of `ex1_dataset.json`
  - first `80%` of `seijou1.json`
- evaluation only:
  - remaining `20%` of the normal half of `ex1_dataset.json`
  - all abnormal half of `ex1_dataset.json`
  - remaining `20%` of `seijou1.json`
  - all of `ijou1.json`

This is stricter than the previous setup because the evaluation metrics come from samples that were not used for model fitting.

Current default API model performance:

```text
Held-out ex1:
accuracy: 0.978
precision: 0.975
recall: 1.000
f1: 0.987

Held-out seijou1 / ijou1:
accuracy: 0.911
precision: 0.949
recall: 0.944
f1: 0.947

Combined held-out evaluation:
accuracy: 0.932
precision: 0.957
recall: 0.962
f1: 0.960
```

For the exhibition demo, the robust Feature-AE remains the better AE-based candidate. It uses robust window-level CSI features instead of trying to reconstruct the noisy raw sequence directly.

The trained model used by the API server is stored at:

```text
models/feature_ae_w500/model.pt
```

The current default artifact is the held-out split model described above.

## Interpretation

This is an anomaly detection setup, not a supervised classifier. The model learns normal CSI behavior from the normal half only. Abnormal detection is based on reconstruction error.

Current `train_eval_features.py` behavior:

- training uses only normal windows
- threshold is set from the validation split of those training-side normal windows
- if you pass only `--data`, training and evaluation come from the same source file
- if you pass `--train-data` and evaluation inputs, training and evaluation are separated by dataset

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

On the exhibition Mac mini:

```sh
docker compose up --build
```

The ML API will listen on:

```text
http://localhost:8001
```

The UI container from `wifi-camera-demo` runs on the same Mac mini. Start it in a separate terminal:

```sh
cd ../wifi-camera-demo
docker compose up --build
```

Then open the web app on the same Mac mini:

```text
http://localhost:3000
```

The web app should use:

```text
/api/ml/latest
```

The UI container proxies that path to `http://host.docker.internal:8001/latest`, so no `ML_API_BASE_URL` override is needed for the one-Mac-mini exhibition setup.

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
  "threshold": 0.5497710901498793,
  "timestamp": "2026-05-27T16:30:00+09:00",
  "samplesBuffered": 500,
  "windowSize": 500,
  "featureNames": ["median", "iqr", "p95_abs"]
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

To stream multiple datasets in sequence, pass `--data` multiple times:

```sh
PYTHONPATH=src python3 -m csi_lstm_ae.simulate_stream \
  --data data/ex1_dataset.json \
  --data /Users/uchimakikohki/Downloads/seijou1.json \
  --data /Users/uchimakikohki/Downloads/ijou1.json \
  --url http://localhost:8001/csi \
  --chunk-size 100 \
  --sleep 1 \
  --sleep-between-datasets 2
```
