"""Standalone evaluation entry point for SGFRNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from dataset import TestSetLoader
from train import (
    PROJECT_ROOT,
    evaluate,
    get_device,
    load_checkpoint,
    make_model,
    print_metrics,
    resolve_path,
)
from utils import seed_pytorch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SGFRNet")
    parser.add_argument("--dataset-name", default="IRSTD-1K")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Default: ./runs/<dataset-name>/SGFRNet_best.pth.tar",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="Examples: cuda, cuda:0, cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_path(args.dataset_dir)
    output_dir = resolve_path(args.output_dir) / args.dataset_name
    checkpoint = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else PROJECT_ROOT / "runs" / args.dataset_name / "SGFRNet_best.pth.tar"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    seed_pytorch(args.seed)
    device = get_device(args.device)
    dataset = TestSetLoader(
        dataset_dir=dataset_dir,
        dataset_name=args.dataset_name,
        split=args.test_split,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = make_model(device)
    load_checkpoint(checkpoint, model, device)
    prediction_dir = output_dir / "predictions" if args.save_predictions else None
    metrics = evaluate(model, loader, device, args.threshold, prediction_dir)
    print_metrics("Test", metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset_name,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT))
        if checkpoint.is_relative_to(PROJECT_ROOT)
        else str(checkpoint),
        "threshold": args.threshold,
        **metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
