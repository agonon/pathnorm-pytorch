from pathlib import Path
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import math

FULL_SPLIT_TRAIN_VAL = 0.99


def get_B(data_dir, batch_size, workers):
    """
    Compute the constant B := maximum L infinity norm of the inputs normalized
    for inference, as prescribed in Section 4 of
    https://arxiv.org/abs/2310.01225.
    Args:
        data_dir (Path): Path to the dataset.
        batch_size (int): Batch size for data loader.
        workers (int): Number of workers for data loader.

    Returns:
        float: The computed constant B.
    """
    print("=> Creating dataloader")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    basic_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_val_augmented = datasets.ImageFolder(
        data_dir / "train", basic_transforms
    )

    trainset, valset = torch.utils.data.random_split(
        train_val_augmented,
        [
            int(FULL_SPLIT_TRAIN_VAL * len(train_val_augmented)),
            len(train_val_augmented)
            - int(FULL_SPLIT_TRAIN_VAL * len(train_val_augmented)),
        ],
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    print("=> Computing B")
    B = 0
    # compute max L infinity norm of the inputs
    for i, (X, _) in enumerate(train_loader):
        if i % 100 == 0 or i == 1:
            if i == 1:
                print(
                    f"batch {i}/{len(train_loader)}, will now print every 100 batches"
                )
            else:
                print(f"batch {i}/{len(train_loader)}")
        B = max(B, X.abs().max().item())
    return B


def main(D, P, K, B, n, din, dout):
    """
    Compute the 1st part of the generalization bound given in Theorem 3.1 of
    https://arxiv.org/abs/2310.01225, and the sharpened version (Remark 3.1).

    Args:
        D (int): Depth of the architecture.
        P (int): Number of distinct types of *-max-pooling neurons in the
        architecture.
        K (int): Maximal kernel size.
        B (float): Maximum L infinity norm of the samples.
        n (int): Number of samples.
        din (int): Input dimension.
        dout (int): Output dimension.
    """
    # 1ST PART BOUND
    term_1 = D * math.log((3 + 2 * P) * K)
    frac = (3 + 2 * P) / (1 + P)
    term_2 = math.log(frac * (din + 1) * dout)
    C = math.sqrt(term_1 + term_2)
    bound = C * B * 4 / math.sqrt(n)

    # 1ST PART SHARPENED BOUND
    M = 1
    t1sharp = D * math.log(3) + M * math.log(K)
    t2sharp = math.log((din + 1) * dout)
    Csharp = math.sqrt(t1sharp + t2sharp)
    sharpened_bound = Csharp * B * 4 / math.sqrt(n)

    # print("t1/t2 = ", term_1/term_2)
    # print("t1sharp/t2sharp = ", t1sharp/t2sharp)
    # print("C = ", C)
    # print("math.sqrt(n) = ", math.sqrt(n))
    print("bound = ", bound)
    print("sharpened bound = ", sharpened_bound)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute 1st part bound for ResNets"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        metavar="DIR",
        nargs="?",
        default=None,
        help="path to dataset (default: imagenet)",
    )

    parser.add_argument(
        "--B",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    # DIMENSIONS RELATED TO IMAGENET

    n = int(FULL_SPLIT_TRAIN_VAL * 1281167)
    din = 224 * 224 * 3
    dout = 1000

    # NORM OF IMAGES IN IMAGENET

    if args.B is None and args.data_dir is not None:
        B = get_B(args.data_dir, args.batch_size, args.workers)
        print("B=", B)
    elif args.B is not None:
        B = args.B
    else:
        raise ValueError("B and data_dir cannot be both None")

    # DIMENSIONS RELATED TO RESNETS ARCHITECTURE

    P = 1
    K = 9

    for depth in [18, 34, 50, 101, 152]:
        print(
            f"--------------------------------resnet{depth}--------------------------------"
        )
        main(depth, P, K, B, n, din, dout)
