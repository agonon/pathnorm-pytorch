from enum import Enum

import torch
from torch import distributed as dist


class Summary(Enum):
    """
    Enumeration for different summary types.

    Attributes:
        NONE (int): No summary.
        AVERAGE (int): Average summary.
        SUM (int): Sum summary.
        COUNT (int): Count summary.
    """

    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """
    Computes and stores the average and current value.

    Args:
        name (str): Name of the meter.
        fmt (str): Format string for displaying values (default is ":f").
        summary_type (Summary): Type of summary to display (default is Summary.AVERAGE).

    Attributes:
        name (str): Name of the meter.
        fmt (str): Format string for displaying values.
        summary_type (Summary): Type of summary to display.
        val (float): Current value.
        avg (float): Average value.
        sum (float): Sum of values.
        count (int): Number of values.

    Methods:
        reset(): Reset the meter values.
        update(val, n=1): Update the meter with a new value.
        all_reduce(): Perform all-reduce operation for distributed training.
        __str__(): String representation of the meter.
        summary(): Summary representation of the meter.
    """

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """
        Reset the meter values.

        Returns:
            None
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): New value.
            n (int): Number of occurrences of the value (default is 1).

        Returns:
            None
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """
        Perform all-reduce operation for distributed training.

        Returns:
            None
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor(
            [self.sum, self.count], dtype=torch.float32, device=device
        )
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        """
        String representation of the meter.

        Returns:
            str: Formatted string representation.
        """
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """
        Summary representation of the meter.

        Returns:
            str: Formatted summary string.
        """
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Displays progress during training.

    Args:
        num_batches (int): Total number of batches.
        meters (list): List of AverageMeter instances.
        prefix (str): Prefix for display.

    Attributes:
        batch_fmtstr (str): Batch format string.
        meters (list): List of AverageMeter instances.
        prefix (str): Prefix for display.

    Methods:
        display(batch): Display progress for a specific batch.
        display_summary(): Display a summary of progress.
        _get_batch_fmtstr(num_batches): Get the batch format string.
    """

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """
        Display progress for a specific batch.

        Args:
            batch (int): Current batch number.

        Returns:
            None
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        """
        Display a summary of progress.

        Returns:
            None
        """
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """
        Get the batch format string.

        Args:
            num_batches (int): Total number of batches.

        Returns:
            str: Batch format string.
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
