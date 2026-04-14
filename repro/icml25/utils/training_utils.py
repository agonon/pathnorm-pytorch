from __future__ import annotations

import time
from enum import Enum

import torch


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            return ""
        if self.summary_type is Summary.AVERAGE:
            return "{name} {avg:.3f}".format(**self.__dict__)
        if self.summary_type is Summary.SUM:
            return "{name} {sum:.3f}".format(**self.__dict__)
        if self.summary_type is Summary.COUNT:
            return "{name} {count:.3f}".format(**self.__dict__)
        raise ValueError(f"invalid summary type {self.summary_type!r}")


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def validate(
    loader,
    model,
    criterion,
    device,
    print_freq,
    prefix_print="Val",
    log_progress=True,
):
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.NONE)
    top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, top1, top5],
        prefix=f"{prefix_print}: ",
    )

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if log_progress and i % print_freq == 0:
                progress.display(i + 1)

    if log_progress:
        progress.display_summary()

    return top1.avg, top5.avg, losses.avg


def get_optimizer(args, model):
    return torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch,
    device,
    scaler,
    args,
    log_progress=True,
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    lr_meter = AverageMeter("Lr", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    model.train()
    end = time.time()
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, lr_meter, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )
    autocast_enabled = device.type == "cuda"

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        losses.update(loss.detach().item(), images.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()
            lr_meter.update(scheduler.get_last_lr()[0])
        else:
            lr_meter.update(optimizer.param_groups[0]["lr"])

        batch_time.update(time.time() - end)
        end = time.time()

        if log_progress and i % args.print_freq == 0:
            progress.display(i + 1)

    return top1.avg, top5.avg, losses.avg
