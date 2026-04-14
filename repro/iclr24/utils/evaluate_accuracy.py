import time
import torch
import torch.distributed as dist
from torch.utils.data import Subset

from repro.iclr24.utils.compute_margin import get_quantile_margins
from repro.iclr24.utils.meters import Summary, AverageMeter, ProgressMeter
from repro.iclr24.utils.pathnorm_compat import compute_path_norms


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values
    of k.

    Args:
        output (torch.Tensor): Model predictions.
        target (torch.Tensor): True labels.
        topk (tuple): Tuple specifying the top-k accuracy values to compute.

    Returns:
        list: List of accuracy values for each specified top-k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(
    val_loader,
    model,
    criterion,
    device,
    gpu,
    print_freq,
    distributed,
    world_size,
    batch_size,
    workers,
    prefix_print="Val",
    log_progress=True,
):
    """
    Evaluate the top-1 accuracy, top-5 accuracy and average loss of a model on
    a given dataset.

    Args:
        val_loader (torch.utils.data.DataLoader): DataLoader of the dataset.
        model (torch.nn.Module): The model to be evaluated.
        criterion (torch.nn.Module): The loss criterion.
        device (torch.device): Device on which the model should be placed.
        gpu (int): GPU index to use for validation.
        print_freq (int): Frequency to print batch progress.
        distributed (bool): Whether to use distributed training.
        world_size (int): Number of processes in distributed training.
        batch_size (int): Batch size for validation.
        workers (int): Number of workers for parallel data loading.
        prefix_print (str): Prefix to be printed during validation.

    Returns:
        float: Top-1 accuracy.
        float: Top-5 accuracy.
        float: Average loss.
    """

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if gpu is not None and torch.cuda.is_available():
                    images = images.cuda(gpu, non_blocking=True)
                    target = target.cuda(gpu, non_blocking=True)
                elif torch.cuda.is_available():
                    images = images.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if log_progress and i % print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader)
        + (
            distributed
            and (
                len(val_loader.sampler) * world_size < len(val_loader.dataset)
            )
        ),
        [batch_time, losses, top1, top5],
        prefix=f"{prefix_print}: ",
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if distributed:
        losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()

    if distributed and (
        len(val_loader.sampler) * world_size < len(val_loader.dataset)
    ):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(
                len(val_loader.sampler) * world_size,
                len(val_loader.dataset),
            ),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    if log_progress:
        progress.display_summary()

    return top1.avg, top5.avg, losses.avg


def test_after_training(
    model,
    test_loader,
    train_loader,
    criterion,
    logger,
    imp_iter,
    device,
    exp_dir,
    gpu,
    epochs,
    arch,
    data_parallel,
    print_freq,
    distributed,
    world_size,
    batch_size,
    workers,
):
    """
    Test the model after training using the best checkpoint.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        criterion (torch.nn.Module): The loss criterion.
        logger: Logger object for recording metrics.
        imp_iter (int): Number of current IMP iteration.
        device (torch.device): Device on which the model should be placed.
        exp_dir (Path): Path to the experiment directory.
        gpu (int): GPU index to use for testing.
        epochs (int): Number of training epochs.
        arch (str): Architecture of the model.
        data_parallel (bool): Whether data parallelism is used.
        print_freq (int): Frequency to print progress.
        distributed (bool): Whether distributed training is used.
        world_size (int): Number of processes in distributed training.
        batch_size (int): Batch size for testing.
        workers (int): Number of workers for parallel data loading.
    """
    log_primary = (gpu is None) or (gpu == 0)

    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()

    cp_dir = exp_dir
    if imp_iter > 0:
        cp_dir = exp_dir / f"imp_iter{imp_iter}"
    if log_primary:
        print(
            f"=> Evaluating metrics of the model with best val top-1 accuracy of current IMP iter ({imp_iter})"
        )
    best_val_ckpt_path = cp_dir / "model_best.pth.tar"
    if log_primary:
        print("=> Loading checkpoint '{}'".format(best_val_ckpt_path))
    if gpu is None:
        best_checkpoint = torch.load(best_val_ckpt_path)
    elif torch.cuda.is_available():
        # Map models to be loaded to specified single gpu.
        loc = "cuda:{}".format(gpu)
        best_checkpoint = torch.load(best_val_ckpt_path, map_location=loc)
    model.load_state_dict(best_checkpoint["state_dict"])
    if log_primary:
        print(
            "checkpoint '{}' loaded (epoch {})".format(
                best_val_ckpt_path, best_checkpoint["epoch"]
            )
        )

    train_top1, train_top5, train_loss = validate(
        train_loader,
        model,
        criterion,
        device,
        gpu,
        print_freq,
        distributed,
        world_size,
        batch_size,
        workers,
        prefix_print="Train",
        log_progress=log_primary,
    )

    test_top1, test_top5, test_loss = validate(
        test_loader,
        model,
        criterion,
        device,
        gpu,
        print_freq,
        distributed,
        world_size,
        batch_size,
        workers,
        prefix_print="Test",
        log_progress=log_primary,
    )
    exponents = [1, 2, 4, 8, 16]
    pathnorms = compute_path_norms(
        model, arch, exponents, device, data_parallel
    )

    if distributed:
        if log_primary:
            print(
                "=> Skipping train-margin quantiles in distributed mode "
                "(the current helper only sees one rank shard)"
            )
        quantile_margins_train = [float("nan")] * 5
    else:
        quantile_margins_train = get_quantile_margins(
            model, train_loader, device
        )

    metrics_dict = {
        "epoch": best_checkpoint["epoch"],
        "train/loss": train_loss,
        "train/acc1": (
            train_top1 if isinstance(train_top1, float) else train_top1.item()
        ),
        "train/acc5": (
            train_top5 if isinstance(train_top5, float) else train_top5.item()
        ),
        "test/loss": test_loss,
        "test/acc1": (
            test_top1 if isinstance(test_top1, float) else test_top1.item()
        ),
        "test/acc5": (
            test_top5 if isinstance(test_top5, float) else test_top5.item()
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
        "imp_iter": imp_iter,
    }
    print(metrics_dict)
    logger.log_step(metrics_dict, epochs)
