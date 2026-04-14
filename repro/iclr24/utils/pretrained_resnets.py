import argparse
from pathlib import Path
import sys

from matplotlib import pyplot as plt
import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro.iclr24.utils.compute_margin import get_all_margins
from repro.iclr24.utils.pathnorm_compat import get_path_norm


def get_trainloader(data_dir, batch_size, workers):
    """
    Create a DataLoader from the path to the data directory. It keeps 99% of
    the dataset for training and 1% for validation.

    Args:
        data_dir (Path): Path to the dataset.
        batch_size (int): Batch size for the DataLoader.
        workers (int): Number of workers for parallel data loading.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the training dataset.
    """
    print(f"=> Creating dataloader")

    inference_normalization = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_val_set = datasets.ImageFolder(
        data_dir / "train", inference_normalization
    )

    FULL_SPLIT_TRAIN_VAL = 0.99

    trainset, valset = torch.utils.data.random_split(
        train_val_set,
        [
            int(FULL_SPLIT_TRAIN_VAL * len(train_val_set)),
            len(train_val_set)
            - int(FULL_SPLIT_TRAIN_VAL * len(train_val_set)),
        ],
    )

    print(f"Compute margins on 99% of ImageNet = {len(trainset)} images")

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )

    return trainloader


def get_model(arch, device):
    """
    Download and return a pre-trained ResNet model.

    Args:
        arch (str): Architecture of the model. Should be one of 'resnet18',
        'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        device (torch.device): Device on which the model should be placed.

    Returns:
        torch.nn.Module: Pre-trained ResNet model.
    """
    print(f"=> Downloading pre-trained {arch}")
    if arch == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        ).to(device)
    elif arch == "resnet34":
        model = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        ).to(device)
    elif arch == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        ).to(device)
    elif arch == "resnet101":
        model = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2
        ).to(device)
    elif arch == "resnet152":
        model = models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V2
        ).to(device)
    return model


def plot_and_save_margins(
    model,
    arch,
    trainloader,
    device,
    color,
    margins_already_computed,
    savedir,
):
    """
    Compute and plot margins for a model on a training dataset.

    Args:
        model (torch.nn.Module): The model.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training
        dataset.
        device (torch.device): Device on which the model should be placed.
        color (str): Color for the histogram plot.
        margins_already_computed (bool): Whether margins are already computed
        and saved.
        savedir (Path): Directory to save the results.
    """
    if margins_already_computed:
        print(f"=> Loading margins from {savedir / 'margins.pt'}")
        margins = torch.load(savedir / "margins.pt")
    else:
        print(f"=> Computing margins and saving in {savedir / 'margins.pt'}")
        margins = get_all_margins(model, trainloader, device)
        torch.save(margins, savedir / "margins.pt")

    # then get the quantiles
    quantile_margins = []
    number_negative_margins = (margins < 0).sum()
    print(f"train top 1={1 - number_negative_margins / len(margins)}")
    sorted_margins = torch.sort(margins).values

    # compute the quantiles by taking the margin
    # at the index corresponding to the quantiles
    # (1-t) * number_negative_margins + t * len(margins)
    for i in [
        number_negative_margins,
        int((2 / 3) * number_negative_margins + (len(margins) - 1) / 3),
        int(0.5 * number_negative_margins + 0.5 * (len(margins) - 1)),
        int((1 / 3) * number_negative_margins + 2 * (len(margins) - 1) / 3),
        len(margins) - 1,
    ]:
        quantile_margins.append(sorted_margins[i].item())

    plt.hist(
        margins.cpu().numpy(),
        bins=100,
        label=f"{arch}, train top 1={1 - number_negative_margins / len(margins):.2f}",
        color=color if color is not None else "b",
    )
    # plt.plot(quantile_margins[0], 0, marker="x", label="0")
    # plt.plot(quantile_margins[1], 0, marker="x", label="1/3")
    # plt.plot(quantile_margins[2], 0, marker="x", label="1/2")
    # plt.plot(quantile_margins[3], 0, marker="x", label="2/3")
    # plt.plot(quantile_margins[4], 0, marker="x", label="1")
    plt.legend()

    plt.savefig(savedir / "margins_hist.pdf")

    print(
        f"q-quantile of the margins for q = e, 2e/3 + 1/3, (e+1)/2, e/3 + 2/3, 1 \n (with e = top-1 error): {quantile_margins}"
    )
    plt.show()
    # plt.close()


def print_and_save_pathnorms(model, arch, device, savedir):
    """
    Print and save path-norms for different exponents of a ResNet model.

    Args:
        model (torch.nn.Module): The model, should be a ResNet for now.
        arch (str): Architecture of the model. Should be one of 'resnet18',
        'resnet34', 'resnet50', 'resnet101', 'resnet152'.
        device (torch.device): Device on which the model should be placed.
        savedir (Path): Directory to save the results.
    """
    exponents = [1, 2, 4, 8, 16]
    pathnorms = []
    for exponent in exponents:
        pathnorm = get_path_norm(
            model,
            arch,
            device,
            exponent=exponent,
            constant_input=1,
            in_place=True,
        )
        print(f"L^{exponent} path-norm = {pathnorm}")
        pathnorms.append(pathnorm)
    pathnorms = torch.tensor(pathnorms)
    torch.save(pathnorms, savedir / "pathnorms.pt")


def main(
    arch,
    device,
    trainloader=None,
    margins_already_computed=False,
    saving_dir=False,
    color=None,
    margins=True,
    pathnorms=True,
):
    model = get_model(arch, device)
    model.eval()

    savedir = saving_dir / f"{arch}"
    savedir.mkdir(parents=True, exist_ok=True)

    if margins:
        if margins_already_computed or trainloader is not None:
            plot_and_save_margins(
                model,
                arch,
                trainloader,
                device,
                color,
                margins_already_computed,
                savedir,
            )
        else:
            print(
                "Margins skipped: please provide path to dataset or call with --margins_already_computed."
            )

    if pathnorms:
        print_and_save_pathnorms(model, arch, device, savedir)


if __name__ == "__main__":
    # add as arg the path to the dataset
    parser = argparse.ArgumentParser(
        description="Compute Pre-trained Pathnorms and Margins"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        metavar="DIR",
        nargs="?",
        default=None,
        help="path to dataset",
    )
    parser.add_argument(
        "--saving-dir",
        type=Path,
        metavar="DIR",
        nargs="?",
        default=None,
        help="path to save pathnorms and margins, and path to load margins if already computed",
    )

    parser.add_argument(
        "--margins-already-computed",
        action="store_true",
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

    if args.margins_already_computed:
        trainloader = None
        print("You choose to use pre-computed margins.")
    else:
        if args.data_dir is not None:
            trainloader = get_trainloader(
                args.data_dir, args.batch_size, args.workers
            )
        else:
            raise ValueError(
                "Please provide dataset directory or call with --margins_already_computed."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    colormap = plt.get_cmap("hot")
    colors = [colormap(i) for i in [0, 0.2, 0.4, 0.6, 0.8]]
    for i, arch in enumerate(
        ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    ):
        print(
            f"--------------------------------{arch}--------------------------------"
        )
        main(
            arch,
            device,
            trainloader=trainloader,
            margins_already_computed=args.margins_already_computed,
            saving_dir=args.saving_dir,
            color=colors[i],
            margins=True,  # set to False if you don't want to compute margins
            pathnorms=True,  # set to False if you don't want to compute pathnorms
        )
