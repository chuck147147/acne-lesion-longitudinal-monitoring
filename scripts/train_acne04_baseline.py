"""Train a standard pretrained YOLO11n single-class ACNE04 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "acne04_yolo" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="acne04_yolo11n_fold0")
    args = parser.parse_args()
    if not DATA.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {DATA}. Run scripts/prepare_acne04_yolo.py first.")
    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA is not available in this Python environment. Use the project .venv CUDA setup.")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=4,
        patience=20,
        seed=42,
        deterministic=True,
        pretrained=True,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
    )

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training ended without best checkpoint: {best}")
    best_model = YOLO(str(best))
    metrics = best_model.val(data=str(DATA), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device, plots=True)
    print("Official test metrics:")
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
