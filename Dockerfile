FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/feature_ae_w500/model.pt
ENV MODEL_DEVICE=cpu

EXPOSE 8001

CMD ["uvicorn", "csi_lstm_ae.api_server:app", "--host", "0.0.0.0", "--port", "8001"]
