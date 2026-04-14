import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re


def save_figure(saving_dir: Path, stem: str) -> None:
    plt.savefig(saving_dir / f"{stem}.pdf")
    plt.savefig(saving_dir / f"{stem}.png", dpi=200)


def size_fraction_label(size_dataset):
    known = {
        39636: "1/32",
        79272: "1/16",
        158544: "1/8",
        317089: "1/4",
        634178: "1/2",
    }
    return known.get(size_dataset, str(size_dataset))


def resolve_results_training_path(results_training_dir, rank):
    requested_path = results_training_dir / f"rank={rank}" / "csv" / "results.csv"
    if requested_path.exists():
        return requested_path

    discovered_paths = sorted(results_training_dir.glob("rank=*/csv/results.csv"))
    if len(discovered_paths) == 1:
        discovered_path = discovered_paths[0]
        print(
            "=> Using discovered results file "
            f"{discovered_path} instead of missing requested rank={rank}"
        )
        return discovered_path

    raise FileNotFoundError(
        f"Could not find results.csv for rank={rank} under {results_training_dir}. "
        f"Discovered candidates: {[str(path) for path in discovered_paths]}"
    )


def main(
    num_epochs,
    num_seeds,
    saving_dir,
    results_training_dir,
    size_datasets,
    rank,
):
    """
    A utility for averaging and visualizing training results based on different
    dataset sizes.

    Args:
        num_epochs (int): Number of epochs.
        num_seeds (int): Number of seeds.
        saving_dir (Path): Directory to save the generated figures.
        results_training_dir (Path): Directory containing the results of the
        training.

    Raises:
        ValueError: If the provided results_training_dir does not contain the
        required patterns.

    Note:
        Assumes that the results_training_dir follows a specific structure with
        placeholders 'seed=x' and 'size_dataset=x'.
    """
    list_train_losses = []
    list_test_losses = []
    list_train_top1 = []
    list_test_top1 = []
    list_pathnorm1 = []
    list_pathnorm2 = []
    list_pathnorm4 = []
    list_epochs = []

    print(f"=> Averaging results from {num_seeds} seeds.")
    for size_dataset in size_datasets:
        for seed in range(num_seeds):
            # look for a pattern seed=x and replace x by the new seed
            pattern = re.compile(r"seed=\d+")
            resolved_training_dir = Path(
                str(results_training_dir).replace(
                    pattern.search(str(results_training_dir)).group(),
                    f"seed={seed}",
                )
            )
            # look for a pattern size_dataset=x and replace x by the new size_dataset
            pattern = re.compile(r"size_dataset=\d+")
            resolved_training_dir = Path(
                str(resolved_training_dir).replace(
                    pattern.search(str(resolved_training_dir)).group(),
                    f"size_dataset={size_dataset}",
                )
            )
            results_training_path = resolve_results_training_path(
                resolved_training_dir, rank
            )
            df = pd.read_csv(results_training_path).copy()
            df = df.dropna(subset=["epoch"])
            # Keep one entry per epoch so the final post-training summary row
            # does not create duplicate trailing points in the plots.
            df = df.drop_duplicates(subset="epoch", keep="first")

            epochs = df["epoch"].values[:num_epochs]
            if seed == 0:
                train_losses = df["train/loss"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                test_losses = df["test/loss"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                train_top1 = df["train/acc1"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                test_top1 = df["test/acc1"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]

                pathnorm1 = df["pathnorm1"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                pathnorm2 = df["pathnorm2"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                pathnorm4 = df["pathnorm4"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
            else:
                train_losses += df["train/loss"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                test_losses += df["test/loss"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                train_top1 += df["train/acc1"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                test_top1 += df["test/acc1"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]

                pathnorm1 += df["pathnorm1"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                pathnorm2 += df["pathnorm2"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
                pathnorm4 += df["pathnorm4"].to_numpy(
                    dtype=float, copy=True
                )[:num_epochs]
        train_losses /= num_seeds
        test_losses /= num_seeds
        train_top1 /= num_seeds
        test_top1 /= num_seeds
        pathnorm1 /= num_seeds
        pathnorm2 /= num_seeds
        pathnorm4 /= num_seeds

        list_train_losses.append(train_losses)
        list_test_losses.append(test_losses)
        list_train_top1.append(train_top1)
        list_test_top1.append(test_top1)
        list_pathnorm1.append(pathnorm1)
        list_pathnorm2.append(pathnorm2)
        list_pathnorm4.append(pathnorm4)
        list_epochs.append(epochs)

    print(f"=> Saving plots in {saving_dir}")

    colormap = plt.get_cmap("hot")
    colors = [
        colormap(i / len(list_train_losses))
        for i in range(len(list_train_losses))
    ]

    # plot top 1 generalization error
    for i, size_dataset in enumerate(size_datasets):
        epochs_i = list_epochs[i]
        # plt.plot(
        #     epochs_i,
        #     100 - list_train_top1[i],
        #     label=f"Train {fraction_size_dataset[size_dataset]}",
        #     color=colors[i],
        # )
        # plt.plot(
        #     epochs_i,
        #     100 - list_test_top1[i],
        #     label=f"Test {fraction_size_dataset[size_dataset]}",
        #     color=colors[i],
        # )
        gene_top_1 = (100 - list_test_top1[i]) - (100 - list_train_top1[i])
        plt.plot(
            epochs_i,
            gene_top_1,
            label=f"{size_fraction_label(size_dataset)}",
            color=colors[i],
        )
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Generalization error for top-1 accuracy (%)")
    save_figure(saving_dir, "top1")
    plt.close()

    # plot cross-entropy generalization error
    for i, size_dataset in enumerate(size_datasets):
        epochs_i = list_epochs[i]
        # plt.plot(
        #     epochs_i,
        #     list_train_losses[i],
        #     label=f"Train {fraction_size_dataset[size_dataset]}",
        #     color=colors[i],
        # )
        # plt.plot(
        #     epochs_i,
        #     list_test_losses[i],
        #     label=f"Test {fraction_size_dataset[size_dataset]}",
        #     color=colors[i],
        # )
        generalization_error_CE = list_test_losses[i] - list_train_losses[i]
        plt.plot(
            epochs_i,
            generalization_error_CE,
            label=f"{size_fraction_label(size_dataset)}",
            color=colors[i],
        )

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Generalization error for the cross-entropy loss")
    save_figure(saving_dir, "cross_entropy")
    plt.close()

    # plot L1 path-norm
    for i, size_dataset in enumerate(size_datasets):
        epochs_i = list_epochs[i]
        plt.plot(
            epochs_i,
            list_pathnorm1[i],
            label=f"{size_fraction_label(size_dataset)}",
            color=colors[i],
        )

    plt.legend(loc="best")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L1 path-norm")
    save_figure(saving_dir, "L1_path_norm")
    plt.close()

    # plot L1, L2, L4 path-norms
    for i, size_dataset in enumerate(size_datasets):
        epochs_i = list_epochs[i]
        plt.plot(
            epochs_i,
            list_pathnorm1[i],
            label=f"L1 {size_fraction_label(size_dataset)}",
            color=colors[i],
        )
        plt.plot(
            epochs_i,
            list_pathnorm2[i],
            label=f"L2 {size_fraction_label(size_dataset)}",
            color=colors[i],
        )
        plt.plot(
            epochs_i,
            list_pathnorm4[i],
            label=f"L4 {size_fraction_label(size_dataset)}",
            color=colors[i],
        )
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Path-norm")
    save_figure(saving_dir, "path_norms")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=90)
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=3,
        help="Will plot the average results of 4_train_increasing_dataset.sh over all integer seeds in [0, num_seeds).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=-1,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--saving-dir",
        type=Path,
        default=Path("repro/iclr24/results/rerun/5_plot_increasing_dataset/"),
    )
    parser.add_argument(
        "--results-training-dir",
        type=Path,
        default=None,
        help="Saving directory used in script 4_train_increasing_dataset.sh for an arbitrary seed: where the results of training have been saved.",
    )
    parser.add_argument(
        "--size-datasets",
        type=int,
        nargs="+",
        default=[39636, 79272, 158544, 317089, 634178],
        help="Dataset sizes to include in the plots.",
    )
    args = parser.parse_args()

    if args.results_training_dir is None:
        raise ValueError("Please provide --results-training-dir.")
    pattern = re.compile(r"seed=\d+")
    if not pattern.search(str(args.results_training_dir)):
        raise ValueError(
            "--results-training-dir must be a directory that contains the string 'seed=x', where x is an arbitrary integer. x will be replaced with all integers in [0, num_seeds)."
        )

    pattern = re.compile(r"size_dataset=\d+")
    if not pattern.search(str(args.results_training_dir)):
        raise ValueError(
            "--results-training-dir must be a directory that contains the string 'size_dataset=x', where x is an arbitrary integer."
        )

    args.saving_dir.mkdir(parents=True, exist_ok=True)

    main(
        args.num_epochs,
        args.num_seeds,
        args.saving_dir,
        args.results_training_dir,
        args.size_datasets,
        args.rank,
    )
