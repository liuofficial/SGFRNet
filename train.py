"""Train and evaluate SGFRNet for infrared small-target detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import TestSetLoader, TrainSetLoader
from loss import SGFRNetLoss
from metrics import PD_FA, SamplewiseSigmoidMetric, mIoU
from model.SGFRNet import SGFRNet
from utils import seed_pytorch
from warmup_scheduler import GradualWarmupScheduler


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate SGFRNet")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--dataset-name", default="IRSTD-1K")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--train-split", default="trainval")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume optimizer and scheduler states")
    parser.add_argument("--save-predictions", action="store_true")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--begin-test", type=int, default=500)
    parser.add_argument("--every-test", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="Examples: cuda, cuda:0, cpu")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve command-line paths relative to this repository."""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def get_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def main_prediction(outputs: torch.Tensor | Iterable[torch.Tensor]) -> torch.Tensor:
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def make_model(device: torch.device) -> SGFRNet:
    # SGFRNet initializes its fixed Haar filters internally. Do not apply a
    # global convolution reinitializer, which would overwrite those filters.
    return SGFRNet(in_ch=1, base_ch=16, deep_supervision=True).to(device)


def strip_wrapper_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Accept checkpoints saved by wrapper or data-parallel modules."""
    for prefix in ("module.model.", "model.", "module."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(strip_wrapper_prefix(state_dict), strict=True)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_mIoU", 0.0))


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    best_miou: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_mIoU": best_miou,
        },
        path,
    )


def original_size(size) -> Tuple[int, int]:
    def scalar(value) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item())
        if isinstance(value, (list, tuple)):
            return scalar(value[0])
        return int(value)

    return scalar(size[0]), scalar(size[1])


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    prediction_dir: Path | None = None,
) -> Dict[str, float]:
    model.eval()
    pixel_metric = mIoU()
    target_metric = PD_FA()
    sample_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=threshold)

    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image, mask, size, image_id in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            height, width = original_size(size)

            probability = main_prediction(model(image))[:, :, :height, :width]
            mask = mask[:, :, :height, :width]
            prediction = probability >= threshold

            sample_metric.update(probability, mask)
            pixel_metric.update(prediction, mask)
            target_metric.update(prediction[0, 0], mask[0, 0], (height, width))

            if prediction_dir is not None:
                name = image_id[0] if isinstance(image_id, (list, tuple)) else str(image_id)
                output = (prediction[0, 0].cpu().numpy().astype(np.uint8) * 255)
                Image.fromarray(output).save(prediction_dir / f"{name}.png")

    pixel_accuracy, miou_value = pixel_metric.get()
    pd_value, fa_value = target_metric.get()
    return {
        "pixel_accuracy": float(pixel_accuracy),
        "mIoU": float(miou_value),
        "nIoU": float(sample_metric.get()),
        "Pd": float(pd_value),
        "Fa": float(fa_value),
        "Fa_x1e6": float(fa_value * 1e6),
    }


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    print(
        f"{prefix}: mIoU={metrics['mIoU']:.4f}, nIoU={metrics['nIoU']:.4f}, "
        f"Pd={metrics['Pd']:.4f}, Fa(x1e-6)={metrics['Fa_x1e6']:.4f}"
    )


def make_test_loader(args: argparse.Namespace) -> DataLoader:
    dataset = TestSetLoader(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        split=args.test_split,
    )
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)


def run_test(args: argparse.Namespace, device: torch.device) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in test mode")
    model = make_model(device)
    load_checkpoint(args.checkpoint, model, device)
    prediction_dir = args.save_dir / args.dataset_name / "predictions" if args.save_predictions else None
    metrics = evaluate(model, make_test_loader(args), device, args.threshold, prediction_dir)
    print_metrics("Test", metrics)


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    train_dataset = TrainSetLoader(
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        patch_size=args.patch_size,
        split=args.train_split,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= args.batch_size,
    )
    test_loader = make_test_loader(args)

    model = make_model(device)
    criterion = SGFRNetLoss(warm_epoch=40)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.min_learning_rate,
    )
    if args.warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1.0, total_epoch=args.warmup_epochs, after_scheduler=cosine
        )
    else:
        scheduler = cosine

    start_epoch = 0
    best_miou = 0.0
    if args.checkpoint is not None:
        optimizer_to_load = optimizer if args.resume else None
        scheduler_to_load = scheduler if args.resume else None
        loaded_epoch, loaded_best = load_checkpoint(
            args.checkpoint, model, device, optimizer_to_load, scheduler_to_load
        )
        if args.resume:
            start_epoch, best_miou = loaded_epoch, loaded_best

    run_dir = args.save_dir / args.dataset_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.jsonl"

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running = {"total": 0.0, "bce_dice": 0.0, "sls_iou": 0.0}
        sample_count = 0

        for image, mask in train_loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            losses = criterion(model(image), mask, epoch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = image.shape[0]
            sample_count += batch_size
            for key in running:
                running[key] += float(losses[key].detach()) * batch_size

        scheduler.step()
        train_stats = {key: value / max(1, sample_count) for key, value in running.items()}
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], **train_stats}
        print(
            f"Epoch {epoch}/{args.epochs}: loss={train_stats['total']:.6f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

        should_test = epoch >= args.begin_test and epoch % args.every_test == 0
        if should_test or epoch == args.epochs:
            metrics = evaluate(model, test_loader, device, args.threshold)
            print_metrics(f"Epoch {epoch} validation", metrics)
            record.update(metrics)
            if metrics["mIoU"] > best_miou:
                best_miou = metrics["mIoU"]
                save_checkpoint(
                    run_dir / "SGFRNet_best.pth.tar",
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_miou,
                )

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    save_checkpoint(
        run_dir / "SGFRNet_last.pth.tar",
        args.epochs,
        model,
        optimizer,
        scheduler,
        best_miou,
    )


def main() -> None:
    args = parse_args()
    args.dataset_dir = resolve_path(args.dataset_dir)
    args.save_dir = resolve_path(args.save_dir)
    if args.checkpoint is not None:
        args.checkpoint = resolve_path(args.checkpoint)

    seed_pytorch(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset_dir / args.dataset_name}")

    if args.mode == "test":
        run_test(args, device)
    else:
        run_train(args, device)


if __name__ == "__main__":
    main()
