from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send sample CSI data to the ML API as a fake stream.")
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help="Dataset JSON path. Pass multiple times to stream files in sequence.",
    )
    parser.add_argument("--url", default="http://localhost:8001/csi")
    parser.add_argument("--chunk-size", type=int, default=100, help="Samples per POST")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds between POST requests")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--sleep-between-datasets",
        type=float,
        default=0.0,
        help="Optional pause in seconds before switching to the next dataset.",
    )
    return parser.parse_args()


def load_signal(path: str | Path) -> list[float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [float(row[1]) for row in payload["data_matrix"]]


def main() -> None:
    args = parse_args()
    dataset_paths = args.data

    while True:
        for dataset_path in dataset_paths:
            signal = load_signal(dataset_path)
            index = args.start_index
            dataset_name = Path(dataset_path).name

            print(f"dataset={dataset_name} samples={len(signal)}")

            while index < len(signal):
                chunk = signal[index : index + args.chunk_size]
                if not chunk:
                    break

                response = requests.post(
                    args.url,
                    json={
                        "samplingRateHz": 100,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "pc1PhaseVariation": chunk,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
                print(
                    f"dataset={dataset_name} "
                    f"sent={index:05d}-{index + len(chunk) - 1:05d} "
                    f"status={result['status']} "
                    f"score={result['anomalyScore']} "
                    f"error={result['reconstructionError']}"
                )

                index += len(chunk)
                if args.sleep > 0:
                    time.sleep(args.sleep)

            if args.sleep_between_datasets > 0:
                time.sleep(args.sleep_between_datasets)

        if not args.loop:
            break


if __name__ == "__main__":
    main()
