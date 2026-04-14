from __future__ import annotations

import argparse
import random
import time
import warnings
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models

from .imagenet_dataset import get_dataloaders
from .log_and_checkpoint import (
    get_logger,
    log_and_save_checkpoint,
    resume,
)
from .scheduler import get_scheduler
from .training_utils import get_optimizer, train_one_epoch, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICML 2025 dense ImageNet training."
    )
    parser.add_argument("data", type=Path, help="Path to prepared ImageNet.")
    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    )
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--wd",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["constant", "cosine", "multi-step"],
        default="multi-step",
    )
    parser.add_argument("--saving-dir", type=Path, required=True)
    parser.add_argument("--print-freq", type=int, default=400)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--size-dataset", type=int, default=None)
    parser.add_argument("--evaluate-before-train", action="store_true")
    parser.add_argument("--test-after-train", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    warnings.warn(
        "You have chosen to seed training. This enables deterministic "
        "CUDNN settings and may slow training."
    )


def build_model(arch: str, device: torch.device) -> torch.nn.Module:
    print(f"=> creating model '{arch}'")
    return models.__dict__[arch]().to(device)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = get_device()
    model = build_model(args.arch, device)
    criterion = nn.CrossEntropyLoss().to(device)
    train_loader, val_loader, test_loader, _ = get_dataloaders(args)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, len(train_loader))

    best_val_top1 = 0.0
    if args.resume is not None:
        args.start_epoch, best_val_top1 = resume(
            model,
            optimizer,
            scheduler,
            args.resume,
            device,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    logger = get_logger(args.saving_dir, tensorboard=args.tensorboard)

    if args.evaluate_before_train:
        validate(
            test_loader,
            model,
            criterion,
            device,
            args.print_freq,
            prefix_print="Test",
        )

    print("=> Begin ICML dense training")
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        train_top1, train_top5, train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            device,
            scaler,
            args,
        )
        val_top1, val_top5, val_loss = validate(
            val_loader,
            model,
            criterion,
            device,
            args.print_freq,
            prefix_print="Val",
        )
        test_top1, test_top5, test_loss = validate(
            test_loader,
            model,
            criterion,
            device,
            args.print_freq,
            prefix_print="Test",
        )
        best_val_top1 = log_and_save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_top1=val_top1,
            best_val_top1=best_val_top1,
            exp_dir=args.saving_dir,
            arch=args.arch,
            train_loss=train_loss,
            train_top1=train_top1,
            train_top5=train_top5,
            val_loss=val_loss,
            val_top5=val_top5,
            test_loss=test_loss,
            test_top5=test_top5,
            test_top1=test_top1,
            epoch_time=time.time() - epoch_start,
            logger=logger,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    logger.close()
    print(f"Finished dense training under {args.saving_dir}.")


if __name__ == "__main__":
    main()
