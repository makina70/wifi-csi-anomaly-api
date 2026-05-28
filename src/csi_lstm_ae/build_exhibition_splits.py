from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible train/eval JSON splits for exhibition CSI data.")
    parser.add_argument("--ex1", default="data/ex1_dataset.json")
    parser.add_argument("--seijou", required=True)
    parser.add_argument("--ijou", required=True)
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def load_payload(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_payload(path: Path, metadata: dict, rows: list[list[float]]) -> None:
    payload = {
        "metadata": {
            **metadata,
            "matrix_shape": [len(rows), 2],
        },
        "data_matrix": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def normalize_time(rows: list[list[float]], sampling_rate_hz: float) -> list[list[float]]:
    if not rows:
        return rows
    dt = 1.0 / sampling_rate_hz
    return [[round(index * dt, 10), float(row[1])] for index, row in enumerate(rows)]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ex1 = load_payload(args.ex1)
    seijou = load_payload(args.seijou)
    ijou = load_payload(args.ijou)

    ex1_rows = ex1["data_matrix"]
    seijou_rows = seijou["data_matrix"]
    ijou_rows = ijou["data_matrix"]

    sampling_rate_hz = float(ex1.get("metadata", {}).get("sampling_rate_hz", 100.0))
    ex1_normal_end = len(ex1_rows) // 2
    ex1_normal_rows = ex1_rows[:ex1_normal_end]
    ex1_abnormal_rows = ex1_rows[ex1_normal_end:]

    ex1_train_end = int(len(ex1_normal_rows) * args.train_ratio)
    seijou_train_end = int(len(seijou_rows) * args.train_ratio)

    ex1_train_normal = normalize_time(ex1_normal_rows[:ex1_train_end], sampling_rate_hz)
    ex1_eval_normal = normalize_time(ex1_normal_rows[ex1_train_end:], sampling_rate_hz)
    ex1_eval_abnormal = normalize_time(ex1_abnormal_rows, sampling_rate_hz)

    seijou_train_normal = normalize_time(seijou_rows[:seijou_train_end], sampling_rate_hz)
    seijou_eval_normal = normalize_time(seijou_rows[seijou_train_end:], sampling_rate_hz)
    ijou_eval_abnormal = normalize_time(ijou_rows, sampling_rate_hz)

    ex1_train_metadata = {
        "source_file": "ex1_train_normal",
        "sampling_rate_hz": sampling_rate_hz,
        "matrix_columns": ["Time (Seconds)", "PC1_Phase_Variation"],
        "normal_samples": len(ex1_train_normal),
        "split_role": "train_normal",
        "train_ratio": args.train_ratio,
    }
    seijou_train_metadata = {
        "source_file": "seijou1_train_normal",
        "sampling_rate_hz": sampling_rate_hz,
        "matrix_columns": ["Time (Seconds)", "PC1_Phase_Variation"],
        "normal_samples": len(seijou_train_normal),
        "split_role": "train_normal",
        "train_ratio": args.train_ratio,
    }
    ex1_eval_metadata = {
        "source_file": "ex1_eval_mixed",
        "sampling_rate_hz": sampling_rate_hz,
        "matrix_columns": ["Time (Seconds)", "PC1_Phase_Variation"],
        "normal_samples": len(ex1_eval_normal),
        "split_role": "eval_mixed",
        "train_ratio": args.train_ratio,
        "normal_source": "ex1_held_out_normal",
        "abnormal_source": "ex1_abnormal_half",
    }
    seijou_eval_metadata = {
        "source_file": "seijou1_eval_normal",
        "sampling_rate_hz": sampling_rate_hz,
        "matrix_columns": ["Time (Seconds)", "PC1_Phase_Variation"],
        "normal_samples": len(seijou_eval_normal),
        "split_role": "eval_normal",
        "train_ratio": args.train_ratio,
    }
    ijou_eval_metadata = {
        "source_file": "ijou1_eval_abnormal",
        "sampling_rate_hz": sampling_rate_hz,
        "matrix_columns": ["Time (Seconds)", "PC1_Phase_Variation"],
        "normal_samples": 0,
        "split_role": "eval_abnormal",
        "train_ratio": args.train_ratio,
    }

    write_payload(out_dir / "ex1_train_normal.json", ex1_train_metadata, ex1_train_normal)
    write_payload(out_dir / "seijou1_train_normal.json", seijou_train_metadata, seijou_train_normal)
    write_payload(out_dir / "ex1_eval_mixed.json", ex1_eval_metadata, ex1_eval_normal + ex1_eval_abnormal)
    write_payload(out_dir / "seijou1_eval_normal.json", seijou_eval_metadata, seijou_eval_normal)
    write_payload(out_dir / "ijou1_eval_abnormal.json", ijou_eval_metadata, ijou_eval_abnormal)

    summary = {
        "out_dir": str(out_dir.resolve()),
        "train_ratio": args.train_ratio,
        "counts": {
            "ex1_train_normal_samples": len(ex1_train_normal),
            "ex1_eval_normal_samples": len(ex1_eval_normal),
            "ex1_eval_abnormal_samples": len(ex1_eval_abnormal),
            "seijou1_train_normal_samples": len(seijou_train_normal),
            "seijou1_eval_normal_samples": len(seijou_eval_normal),
            "ijou1_eval_abnormal_samples": len(ijou_eval_abnormal),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
