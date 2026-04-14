import argparse
from pathlib import Path
import torchvision.models as models

model_names = [
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(models.__dict__[name])
]
model_names = sorted(model_names)


def get_args():
    """
    Parse command-line arguments for PyTorch ImageNet training.

    Returns:
        argparse.Namespace: A namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # Dataset
    parser.add_argument(
        "data",
        metavar="DIR",
        nargs="?",
        default="imagenet",
        help="path to dataset (default: imagenet)",
    )
    parser.add_argument("--dummy", action="store_true", help="use fake data")
    parser.add_argument(
        "--blurred", action="store_true", help="use blurred images of ImageNet"
    )

    # Model
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained weights",
    )

    # Training
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers per GPU (default: 4)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )

    # Optimizer
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["constant", "cosine", "multi-step"],
    )
    parser.add_argument("--clip-grad-norm", type=float, default=None)

    # Augmentations
    parser.add_argument("--mixup-alpha", type=float, default=None)
    parser.add_argument(
        "--random-augmentation-magnitude", type=int, default=None
    )

    # Evaluation
    parser.add_argument(
        "--evaluate-before-train",
        action="store_true",
        help="evaluate models on testing set before training",
    )
    parser.add_argument(
        "--test-after-train",
        action="store_true",
        help="evaluate on test set after training",
    )

    # Seed, checkpoint, logging, output of experiment
    parser.add_argument(
        "--saving-dir",
        type=Path,
        help="directory to save checkpoints and logged metrics",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="log tensorboard"
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=400,
        type=int,
        metavar="N",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="seed for initializing training",
    )

    # Distributed computing parameters
    parser.add_argument("--no-data-parallel", action="store_true")

    parser.add_argument(
        "--device-ids",
        nargs="+",
        type=int,
        default=None,
        help="Devices to use if DataParallel",
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank",
        default=-1,
        type=int,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    # IMP parameters
    parser.add_argument("--IMP-iters", type=int, default=0)
    parser.add_argument("--percentage-pruning", type=float, default=None)
    parser.add_argument("--start-IMP-iter", type=int, default=0)

    # Size of dataset
    parser.add_argument("--size-dataset", type=int, default=None)

    args = parser.parse_args()
    return args
