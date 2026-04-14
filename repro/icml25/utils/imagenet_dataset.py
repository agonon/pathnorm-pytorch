from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

FULL_SPLIT_TRAIN_VAL = 0.99


def get_imagenet_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    basic_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    augmentation_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return basic_transforms, augmentation_transforms


def get_imagenet_train_val_test(imagenet_data: str):
    print(f"=> Getting data from {imagenet_data}")
    traindir = f"{imagenet_data}/train"
    valdir = f"{imagenet_data}/val"
    basic_transforms, augmentation_transforms = get_imagenet_transforms()

    print("=> Creating datasets")
    train_val_augmented = datasets.ImageFolder(traindir, augmentation_transforms)
    trainset, valset = torch.utils.data.random_split(
        train_val_augmented,
        [
            int(FULL_SPLIT_TRAIN_VAL * len(train_val_augmented)),
            len(train_val_augmented)
            - int(FULL_SPLIT_TRAIN_VAL * len(train_val_augmented)),
        ],
        generator=torch.Generator().manual_seed(0),
    )
    testset = datasets.ImageFolder(valdir, basic_transforms)
    return trainset, valset, testset


def get_dataloaders(args: SimpleNamespace):
    if getattr(args, "dummy", False):
        print("=> Dummy data is used!")
        trainset = datasets.FakeData(
            1281167,
            (3, 224, 224),
            1000,
            transforms.ToTensor(),
        )
        valset = datasets.FakeData(
            50000,
            (3, 224, 224),
            1000,
            transforms.ToTensor(),
        )
        testset = datasets.FakeData(
            50000,
            (3, 224, 224),
            1000,
            transforms.ToTensor(),
        )
    else:
        trainset, valset, testset = get_imagenet_train_val_test(args.data)

    if getattr(args, "size_dataset", None) is not None:
        rng = np.random.default_rng(0)
        subset_indices = rng.choice(
            len(trainset),
            args.size_dataset,
            replace=False,
        )
        trainset = torch.utils.data.Subset(trainset, subset_indices)
    else:
        print(
            "Full dataset. Train set is split into "
            f"{FULL_SPLIT_TRAIN_VAL} / {1 - FULL_SPLIT_TRAIN_VAL} "
            "for training and validation."
        )

    print(f"trainset size: {len(trainset)}")
    print(f"valset size: {len(valset)}")
    print(f"testset size: {len(testset)}")

    distributed = getattr(args, "distributed", False)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            valset,
            shuffle=False,
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testset,
            shuffle=False,
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    print("=> Creating dataloaders")
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )
    return train_loader, val_loader, test_loader, train_sampler
