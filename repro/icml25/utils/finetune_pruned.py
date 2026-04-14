from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from .imagenet_dataset import get_dataloaders
from .log_and_checkpoint import get_logger, log_and_save_checkpoint
from .pipeline import (
    DEFAULT_ARCH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_REWIND_EPOCH,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WORKERS,
    build_resnet18,
    epoch5_checkpoint_path,
    load_checkpoint,
    maybe_prepare_pruned_buffers,
    pruned_checkpoint_path,
    rewind_pruned_model,
)
from .scheduler import get_scheduler
from .training_utils import get_optimizer, train_one_epoch, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a one-shot pruned ResNet-18 for the ICML 2025 reproduction."
    )
    parser.add_argument("data", type=Path, help="Path to the prepared ImageNet directory.")
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--rewind-epoch", type=int, default=DEFAULT_REWIND_EPOCH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--wd", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default=DEFAULT_LR_SCHEDULER,
        choices=["constant", "cosine", "multi-step"],
    )
    parser.add_argument("--print-freq", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_training_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        data=str(args.data),
        dummy=False,
        size_dataset=None,
        distributed=False,
        batch_size=args.batch_size,
        workers=args.workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        epochs=args.epochs,
        lr_scheduler=args.lr_scheduler,
        print_freq=args.print_freq,
        seed=args.seed,
        tensorboard=args.tensorboard,
        saving_dir=args.run_dir,
        arch=DEFAULT_ARCH,
    )


def main() -> None:
    args = parse_args()
    training_args = build_training_args(args)
    log_path = args.run_dir / "rank=0" / "csv" / "results.csv"
    if log_path.exists() and not args.overwrite:
        print(f"Training log already exists at {log_path}.")
        return

    pruned_path = pruned_checkpoint_path(args.run_dir)
    rewind_path = epoch5_checkpoint_path(args.dense_dir)
    if not pruned_path.exists():
        raise FileNotFoundError(f"Missing pruned checkpoint at `{pruned_path}`.")
    if not rewind_path.exists():
        raise FileNotFoundError(f"Missing rewind checkpoint at `{rewind_path}`.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pruned_checkpoint = load_checkpoint(pruned_path, device)
    rewind_checkpoint = load_checkpoint(rewind_path, device)

    model = build_resnet18(device)
    maybe_prepare_pruned_buffers(model, pruned_checkpoint["state_dict"])
    model.load_state_dict(pruned_checkpoint["state_dict"])
    rewind_pruned_model(model, rewind_checkpoint["state_dict"])

    criterion = nn.CrossEntropyLoss().to(device)
    train_loader, val_loader, test_loader, _ = get_dataloaders(training_args)
    optimizer = get_optimizer(training_args, model)
    scheduler = get_scheduler(training_args, optimizer, len(train_loader))
    start_epoch = int(rewind_checkpoint.get("epoch", args.rewind_epoch))

    if "optimizer" in rewind_checkpoint:
        # Pruning reparameterizes weights as `weight_orig` + mask buffers,
        # which makes the dense optimizer state unreliable to restore by
        # parameter order. Rewind the weights and scheduler position, but use
        # a fresh optimizer for stable fine-tuning.
        print(
            "Skipping rewind optimizer state for pruned fine-tuning; "
            "continuing with a fresh optimizer."
        )
    if (
        scheduler is not None
        and "scheduler" in rewind_checkpoint
        and rewind_checkpoint["scheduler"] is not None
    ):
        try:
            scheduler.load_state_dict(rewind_checkpoint["scheduler"])
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(
                "Warning: failed to load rewind scheduler state "
                f"({exc}). Continuing with a fresh scheduler."
            )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=torch.cuda.is_available()
    )
    logger = get_logger(args.run_dir, tensorboard=args.tensorboard)

    print(f"=> Begin ICML fine-tuning from epoch {start_epoch}")
    best_val_top1 = 0.0
    for epoch in range(start_epoch, args.epochs):
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
            training_args,
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
            exp_dir=args.run_dir,
            arch=DEFAULT_ARCH,
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
            weight_decay=args.wd,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    logger.close()
    print(f"Finished fine-tuning under {args.run_dir}.")


if __name__ == "__main__":
    main()
