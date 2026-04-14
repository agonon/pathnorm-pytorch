from pathlib import Path

import pandas as pd

import torch.utils.tensorboard

from datetime import datetime

import os
import torch
import shutil
from torch.nn.utils import prune

########### CHECKPOINTING ############


def _atomic_torch_save(state, path):
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_copyfile(src, dst):
    tmp_path = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_checkpoint(
    state, is_best, dir, filename="checkpoint.pth.tar", imp_iter=-1
):
    """
    Save the model if it is the new best epoch or if it is epoch 5 of first IMP
    iteration (for rewinding later on).

    Args:
        state (dict): Model state dictionary.
        is_best (bool): Whether the model is the best so far.
        dir (Path): Directory to save the checkpoint.
        filename (str): Name of the checkpoint file.
        imp_iter (int): Important iteration number.

    Returns:
        None
    """
    if imp_iter > 0:
        dir = dir / f"imp_iter{imp_iter}"
    dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = dir / filename
    _atomic_torch_save(state, checkpoint_path)
    if is_best:
        _atomic_copyfile(checkpoint_path, dir / "model_best.pth.tar")
    if imp_iter == 0 and state["epoch"] == 5:
        # for rewind at epoch 5
        _atomic_copyfile(checkpoint_path, dir / "epoch_5.pth.tar")


def save_init_checkpoint(
    model,
    optimizer,
    scheduler,
    ngpus_per_node,
    multiprocessing_distributed,
    rank,
    best_val_top1,
    arch,
    exp_dir,
):
    """
    Save the model at initialization.

    Args:
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler: Learning rate scheduler.
        ngpus_per_node (int): Number of GPUs per node.
        multiprocessing_distributed (bool): Whether to use multiprocessing for distributed training.
        rank (int): Rank of the current process.
        best_val_top1 (float): Best top-1 validation accuracy.
        arch (str): Model architecture.
        exp_dir (Path): Path to the experiment directory.

    Returns:
        None
    """
    if not multiprocessing_distributed or (
        multiprocessing_distributed and rank % ngpus_per_node == 0
    ):
        is_best = False
        save_checkpoint(
            {
                "epoch": 0,
                "arch": arch,
                "state_dict": model.state_dict(),
                "best_val_top1": best_val_top1,
                "optimizer": optimizer.state_dict(),
                "scheduler": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
            },
            is_best,
            dir=exp_dir,
            filename="init.pth.tar",
        )


def resume(model, optimizer, scheduler, resume, gpu, start_IMP_iter):
    """
    Resume training from a checkpoint.

    Args:
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler: Learning rate scheduler.
        resume (str): Path to the checkpoint file.
        gpu (int): GPU index.
        start_IMP_iter (int): Starting important iteration.

    Returns:
        tuple: Tuple containing the model, optimizer, scheduler, corresponding epoch, and best top-1 accuracy.
    """
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        if gpu is None:
            checkpoint = torch.load(resume)
        elif torch.cuda.is_available():
            # Map models to be loaded to specified single gpu.
            loc = "cuda:{}".format(gpu)
            checkpoint = torch.load(resume, map_location=loc)
        corresponding_epoch = checkpoint["epoch"]
        best_val_top1 = checkpoint["best_val_top1"]
        if gpu is not None:
            # best_val_top1 may be from a checkpoint from a different GPU
            best_val_top1 = best_val_top1.to(gpu)
        if start_IMP_iter > 0:
            # prune once to change the dict
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name="weight", amount=0)
                elif isinstance(module, torch.nn.Linear):
                    # prune only half for the last layer
                    prune.l1_unstructured(
                        module,
                        name="weight",
                        amount=0,
                    )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                resume, checkpoint["epoch"]
            )
        )
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    return model, optimizer, scheduler, corresponding_epoch, best_val_top1


########### LOGGING ############


class Logger:
    def __init__(
        self, metrics_name, csv_dir, tensorboard_dir=None, verbose=False
    ):
        """
        Initialize a Logger.

        Args:
            metrics_name (list): List of metric names.
            csv_dir (Path): Directory to save CSV logs.
            tensorboard_dir (Path): Directory to save TensorBoard logs.
            verbose (bool): Whether to print logs to console.
        """
        # now = datetime.now()
        # dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.metrics_name = metrics_name
        # (csv_dir / dt_string).mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        # self.csv_stats = PandasStats(
        #     csv_dir / dt_string / "results.csv", metrics_name
        # )
        self.csv_stats = PandasStats(csv_dir / "results.csv", metrics_name)
        if tensorboard_dir is not None:
            # self.writer = torch.utils.tensorboard.SummaryWriter(
            #     tensorboard_dir / dt_string
            # )
            self.writer = torch.utils.tensorboard.SummaryWriter(
                tensorboard_dir
            )
        else:
            self.writer = None
        self.verbose = verbose

    def log_step(self, metrics_dict, step):
        """
        Log metrics for a training step.

        Args:
            metrics_dict (dict): Dictionary of metrics.
            step (int): Current step.

        Returns:
            None
        """
        for k in self.metrics_name:
            assert k in self.metrics_name
        dict_for_csv = dict([(k, None) for k in self.metrics_name])
        dict_for_csv.update(metrics_dict)
        self.csv_stats.update(dict_for_csv)
        if self.writer is not None:
            for k, v in metrics_dict.items():
                if v is not None:
                    self.writer.add_scalar(k, v, step)

        if self.verbose:
            print_str = "\t".join(
                [f"{k} {v:.4f}" for k, v in metrics_dict.items()]
            )
            print(print_str)

    def close(self):
        """
        Close the logger.

        Returns:
            None
        """
        if self.writer is not None:
            self.writer.close()


class PandasStats:
    def __init__(self, csv_path, columns):
        """
        Initialize PandasStats.

        Args:
            csv_path (Path): Path to the CSV file.
            columns (list): List of column names.
        """
        self.path = Path(csv_path)
        self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        """
        Update statistics with a new row.

        Args:
            row (dict): New row of data.
            save (bool): Whether to save the statistics.

        Returns:
            None
        """
        self.stats.loc[len(self.stats.index)] = row
        if save:
            self.stats.to_csv(self.path)

    def append(self, df, save=True):
        """
        Append new statistics.

        Args:
            df (pd.DataFrame): DataFrame to append.
            save (bool): Whether to save the statistics.

        Returns:
            None
        """
        self.stats = self.stats.append(df)
        if save:
            self.stats.to_csv(self.path)


def get_logger(exp_dir, rank, tensorboard):
    """
    Get a Logger object.

    Args:
        exp_dir (Path): Path to the experiment directory.
        rank (int): Rank of the current process.
        tensorboard (bool): Whether to use TensorBoard.

    Returns:
        Logger: Logger object.
    """
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
        "pathnorm1",
        "pathnorm2",
        "pathnorm4",
        "pathnorm8",
        "pathnorm16",
        "margin0_train",
        "margin1/3_train",
        "margin1/2_train",
        "margin2/3_train",
        "margin1_train",
        "epoch",
        "epoch time",
        "batch_size",
        "weight_decay",
        "lr",
        "memory",
        "seed",
        "imp_iter",
    ]
    csv_dir = exp_dir / f"rank={rank}" / "csv"
    if tensorboard:
        tensorboard_dir = exp_dir / f"rank={rank}" / "tensorboard"
    else:
        tensorboard_dir = None
    logger = Logger(
        metrics_name, csv_dir=csv_dir, tensorboard_dir=tensorboard_dir
    )
    return logger


########### LOGGING AND CHECKPOINTING ############


def log_and_save_checkpoint(
    val_top1,
    ngpus_per_node,
    epoch,
    model,
    optimizer,
    scheduler,
    train_loss,
    train_top1,
    train_top5,
    val_loss,
    val_top5,
    test_loss,
    test_top5,
    test_top1,
    pathnorms,
    quantile_margins_train,
    best_val_top1,
    epoch_time,
    logger,
    imp_iter,
    multiprocessing_distributed,
    rank,
    arch,
    exp_dir,
    lr,
    weight_decay,
    batch_size,
    seed,
):
    """
    Log metrics and save checkpoint when needed (when model is the new best
    or when epoch 5 of first imp iteration, to allow for rewinding later on).

    Args:
        val_top1 (float): Top-1 accuracy on validation set.
        ngpus_per_node (int): Number of GPUs per node.
        epoch (int): Current epoch.
        model (torch.nn.Module): The model.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler: Learning rate scheduler.
        train_loss (float): Training loss.
        train_top1 (float): Top-1 accuracy on training set.
        train_top5 (float): Top-5 accuracy on training set.
        val_loss (float): Validation loss.
        val_top5 (float): Top-5 accuracy on validation set.
        test_loss (float): Test loss.
        test_top5 (float): Top-5 accuracy on test set.
        test_top1 (float): Top-1 accuracy on test set.
        pathnorms (list): List of path norms.
        quantile_margins_train (list): List of quantile margins on training set.
        best_val_top1 (float): Best top-1 validation accuracy.
        epoch_time (float): Time taken for the epoch.
        logger (Logger): Logger object.
        imp_iter (int): Important iteration.
        multiprocessing_distributed (bool): Whether distributed training is used.
        rank (int): Rank of the current process.
        arch (str): Model architecture.
        exp_dir (Path): Path to the experiment directory.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        batch_size (int): Batch size.
        seed (int): Random seed.

    Returns:
        None
    """
    is_best = val_top1 > best_val_top1
    best_val_top1 = max(val_top1, best_val_top1)

    if not multiprocessing_distributed or (
        multiprocessing_distributed and rank % ngpus_per_node == 0
    ):
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
            is_best,
            dir=exp_dir,
            imp_iter=imp_iter,
        )
        metrics_dict = {
            "train/loss": train_loss,
            "train/acc1": (
                train_top1
                if isinstance(train_top1, float) or isinstance(train_top1, int)
                else train_top1.item()
            ),
            "train/acc5": (
                train_top5
                if isinstance(train_top5, float) or isinstance(train_top5, int)
                else train_top5.item()
            ),
            "test/loss": test_loss,
            "test/acc1": (
                test_top1
                if isinstance(test_top1, float) or isinstance(test_top1, int)
                else test_top1.item()
            ),
            "test/acc5": (
                test_top5
                if isinstance(test_top5, float) or isinstance(test_top5, int)
                else test_top5.item()
            ),
            "val/loss": val_loss,
            "val/acc1": (
                val_top1
                if isinstance(val_top1, float) or isinstance(val_top1, int)
                else val_top1.item()
            ),
            "val/acc5": (
                val_top5
                if isinstance(val_top5, float) or isinstance(val_top5, int)
                else val_top5.item()
            ),
            "pathnorm1": pathnorms[0],
            "pathnorm2": pathnorms[1],
            "pathnorm4": pathnorms[2],
            "pathnorm8": pathnorms[3],
            "pathnorm16": pathnorms[4],
            "margin0_train": quantile_margins_train[0],
            "margin1/3_train": quantile_margins_train[1],
            "margin1/2_train": quantile_margins_train[2],
            "margin2/3_train": quantile_margins_train[3],
            "margin1_train": quantile_margins_train[4],
            "epoch": epoch + 1,
            "epoch time": epoch_time,
            "batch_size": batch_size,
            "lr": scheduler.get_last_lr()[0] if scheduler is not None else lr,
            "memory": torch.cuda.max_memory_allocated() // (1024 * 1024),
            "seed": seed,
            "weight_decay": weight_decay,
            "imp_iter": imp_iter,
        }
        print(metrics_dict)
        logger.log_step(metrics_dict, epoch)
