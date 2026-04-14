from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import torch
import torch.utils.tensorboard


def save_checkpoint(
    state,
    is_best,
    directory,
    filename="checkpoint.pth.tar",
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / filename
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, directory / "model_best.pth.tar")
    if state["epoch"] == 5:
        shutil.copyfile(checkpoint_path, directory / "epoch_5.pth.tar")


def resume(model, optimizer, scheduler, resume_path, device):
    resume_path = Path(resume_path)
    if not resume_path.is_file():
        raise FileNotFoundError(f"No checkpoint found at `{resume_path}`.")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    print(
        f"=> Loaded checkpoint `{resume_path}` "
        f"(epoch {checkpoint['epoch']})"
    )
    return checkpoint["epoch"], checkpoint.get("best_val_top1", 0.0)


class PandasStats:
    def __init__(self, csv_path, columns):
        self.path = Path(csv_path)
        self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row
        if save:
            self.stats.to_csv(self.path)


class Logger:
    def __init__(self, metrics_name, csv_dir, tensorboard_dir=None):
        csv_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_name = metrics_name
        self.csv_stats = PandasStats(csv_dir / "results.csv", metrics_name)
        if tensorboard_dir is not None:
            self.writer = torch.utils.tensorboard.SummaryWriter(tensorboard_dir)
        else:
            self.writer = None

    def log_step(self, metrics_dict, step):
        dict_for_csv = {key: None for key in self.metrics_name}
        dict_for_csv.update(metrics_dict)
        self.csv_stats.update(dict_for_csv)
        if self.writer is not None:
            for key, value in metrics_dict.items():
                if value is not None:
                    self.writer.add_scalar(key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def get_logger(exp_dir, tensorboard):
    metrics_name = [
        "train/loss",
        "train/acc1",
        "train/acc5",
        "val/loss",
        "val/acc1",
        "val/acc5",
        "test/loss",
        "test/acc1",
        "test/acc5",
        "epoch",
        "epoch time",
        "batch_size",
        "weight_decay",
        "lr",
        "memory",
        "seed",
    ]
    csv_dir = Path(exp_dir) / "rank=0" / "csv"
    tensorboard_dir = (
        Path(exp_dir) / "rank=0" / "tensorboard"
        if tensorboard
        else None
    )
    return Logger(
        metrics_name,
        csv_dir=csv_dir,
        tensorboard_dir=tensorboard_dir,
    )


def log_and_save_checkpoint(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    val_top1,
    best_val_top1,
    exp_dir,
    arch,
    train_loss,
    train_top1,
    train_top5,
    val_loss,
    val_top5,
    test_loss,
    test_top5,
    test_top1,
    epoch_time,
    logger,
    lr,
    weight_decay,
    batch_size,
    seed,
):
    is_best = val_top1 > best_val_top1
    best_val_top1 = max(val_top1, best_val_top1)

    save_checkpoint(
        {
            "epoch": epoch + 1,
            "arch": arch,
            "state_dict": model.state_dict(),
            "best_val_top1": best_val_top1,
            "optimizer": optimizer.state_dict(),
            "scheduler": (
                scheduler.state_dict() if scheduler is not None else None
            ),
        },
        is_best=is_best,
        directory=exp_dir,
    )

    metrics_dict = {
        "train/loss": train_loss,
        "train/acc1": train_top1,
        "train/acc5": train_top5,
        "test/loss": test_loss,
        "test/acc1": test_top1,
        "test/acc5": test_top5,
        "val/loss": val_loss,
        "val/acc1": val_top1,
        "val/acc5": val_top5,
        "epoch": epoch + 1,
        "epoch time": epoch_time,
        "batch_size": batch_size,
        "lr": scheduler.get_last_lr()[0] if scheduler is not None else lr,
        "memory": (
            torch.cuda.max_memory_allocated() // (1024 * 1024)
            if torch.cuda.is_available()
            else 0
        ),
        "seed": seed,
        "weight_decay": weight_decay,
    }
    print(metrics_dict)
    logger.log_step(metrics_dict, epoch)
    return best_val_top1
