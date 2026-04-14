import torch.utils.data
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torch.utils.data.dataloader import default_collate

from torchvision.transforms import InterpolationMode

from repro.iclr24.utils.mixup import RandomMixup

NUM_CLASSES = 1000


def get_dataloaders(args):
    if args.dummy:
        print("=> Dummy data is used!")
        trainset = datasets.FakeData(
            1281167, (3, 224, 224), 1000, transforms.ToTensor()
        )
        valset = datasets.FakeData(
            50000, (3, 224, 224), 1000, transforms.ToTensor()
        )
        testset = datasets.FakeData(
            50000, (3, 224, 224), 1000, transforms.ToTensor()
        )
    else:
        trainset, valset, testset = get_imagenet_train_val_test(
            args.data,
            args.blurred,
            randAugLevel=args.random_augmentation_magnitude,
        )

    if args.size_dataset is not None:
        np.random.seed(0)
        # generate args.size_dataset random indices
        subset_indices = np.random.choice(
            len(trainset), args.size_dataset, replace=False
        )
        trainset = torch.utils.data.Subset(trainset, subset_indices)
    else:
        print(
            f"Full dataset. Train set is splitted into {FULL_SPLIT_TRAIN_VAL} / {1 - FULL_SPLIT_TRAIN_VAL} for training and validation."
        )
    print(f"trainset size: {len(trainset)}")
    print(f"valset size: {len(valset)}")
    print(f"testset size: {len(testset)}")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            trainset
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            valset, shuffle=False
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            testset, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    print(f"=> Creating dataloaders")

    if args.mixup_alpha is not None:
        mixup = RandomMixup(NUM_CLASSES, p=1.0, alpha=args.mixup_alpha)

        def collate_fn(batch):
            return mixup(*default_collate(batch))

    else:
        collate_fn = None
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=test_sampler,
    )
    return train_loader, val_loader, test_loader, train_sampler


from torchvision import datasets as datasets

FULL_SPLIT_TRAIN_VAL = 0.99

SPLIT_TRAIN_VAL = 0.9
TOTAL = 1331167
TOTAL_BLURRED = 1331063

SEED_FOR_SPLITTING = 0


def get_imagenet_train_val_test(
    imagenet_data,
    blurred,
    randAugLevel=None,
):
    # Data loading code
    print(f"=> Getting data from {imagenet_data}")
    traindir = os.path.join(
        imagenet_data, "train_blurred" if blurred else "train"
    )
    valdir = os.path.join(imagenet_data, "val_blurred" if blurred else "val")
    basic_transforms, augmentation_transforms = get_imagenet_transforms(
        randAugLevel
    )

    print(f"=> Creating datasets")

    train_val_augmented = datasets.ImageFolder(
        traindir, augmentation_transforms
    )

    trainset, valset = torch.utils.data.random_split(
        train_val_augmented,
        [
            int(FULL_SPLIT_TRAIN_VAL * len(train_val_augmented)),
            len(train_val_augmented)
            - int(FULL_SPLIT_TRAIN_VAL * len(train_val_augmented)),
        ],
    )

    testset = datasets.ImageFolder(valdir, basic_transforms)

    return trainset, valset, testset


def get_imagenet_transforms(randAugLevel):
    """return basic and augmented transforms"""
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
    if randAugLevel is not None:
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    interpolation=InterpolationMode.BILINEAR,
                    magnitude=randAugLevel,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        augmentation_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return basic_transforms, augmentation_transforms
