import argparse
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np


def save_figure(saving_dir: Path, stem: str) -> None:
    plt.savefig(saving_dir / f"{stem}.pdf")
    plt.savefig(saving_dir / f"{stem}.png", dpi=200)


def main(args):
    """
    A utility for averaging and visualizing training results after IMP.

    Args:
        args (argparse.Namespace): Command-line arguments.
            - results-training-dir (Path): Path to the directory containing the training results.
            - saving-dir (Path): Directory to save the generated figures.
            - rank (int): Node rank for distributed training.
            - num-seeds (int): Number of seeds.
    """

    print(f"=> Averaging results from {args.num_seeds} seeds.")
    for seed in range(args.num_seeds):

        results_training_path = (
            args.results_training_dir
            / f"rank={args.rank}"
            / "csv"
            / "results.csv"
        )
        # look for a pattern seed=x and replace x by the new seed
        pattern = re.compile(r"seed=\d+")
        results_training_path = Path(
            str(results_training_path).replace(
                pattern.search(str(results_training_path)).group(),
                f"seed={seed}",
            )
        )
        df = pd.read_csv(results_training_path)

        epochs = df["epoch"].values

        if seed == 0:
            train_losses = df["train/loss"].to_numpy(dtype=float, copy=True)
            test_losses = df["test/loss"].to_numpy(dtype=float, copy=True)
            train_top1 = df["train/acc1"].to_numpy(dtype=float, copy=True)
            test_top1 = df["test/acc1"].to_numpy(dtype=float, copy=True)

            pathnorm1 = df["pathnorm1"].to_numpy(dtype=float, copy=True)
            pathnorm2 = df["pathnorm2"].to_numpy(dtype=float, copy=True)
            pathnorm4 = df["pathnorm4"].to_numpy(dtype=float, copy=True)
            pathnorm8 = df["pathnorm8"].to_numpy(dtype=float, copy=True)
            pathnorm16 = df["pathnorm16"].to_numpy(dtype=float, copy=True)
        else:
            train_losses += df["train/loss"].to_numpy(dtype=float, copy=True)
            test_losses += df["test/loss"].to_numpy(dtype=float, copy=True)
            train_top1 += df["train/acc1"].to_numpy(dtype=float, copy=True)
            test_top1 += df["test/acc1"].to_numpy(dtype=float, copy=True)

            pathnorm1 += df["pathnorm1"].to_numpy(dtype=float, copy=True)
            pathnorm2 += df["pathnorm2"].to_numpy(dtype=float, copy=True)
            pathnorm4 += df["pathnorm4"].to_numpy(dtype=float, copy=True)
            pathnorm8 += df["pathnorm8"].to_numpy(dtype=float, copy=True)
            pathnorm16 += df["pathnorm16"].to_numpy(dtype=float, copy=True)
    # average over seeds
    train_losses /= args.num_seeds
    test_losses /= args.num_seeds
    train_top1 /= args.num_seeds
    test_top1 /= args.num_seeds
    pathnorm1 /= args.num_seeds
    pathnorm2 /= args.num_seeds
    pathnorm4 /= args.num_seeds
    pathnorm8 /= args.num_seeds
    pathnorm16 /= args.num_seeds

    generalization_error_CE = test_losses - train_losses
    generalization_error_top1 = (100 - test_top1) - (100 - train_top1)

    splitting_indices = np.where(np.diff(epochs) < 0)[0] + 1
    list_epochs = np.split(epochs, splitting_indices)

    print(f"=> Saving plots in {args.saving_dir}")

    # plot generalization error cross entropy
    plt.plot(epochs, generalization_error_CE)
    plt.xlabel("Epoch")
    plt.ylabel("Generalization Error (CE)")
    args.saving_dir.mkdir(parents=True, exist_ok=True)
    save_figure(args.saving_dir, "generalization_error_CE")
    plt.close()

    # plot top 1 and generalization error for top 1
    plt.plot(epochs, train_top1, label="train")
    plt.plot(epochs, test_top1, label="test")
    plt.plot(epochs, generalization_error_top1, label="generalization error")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Generalization Error (Top 1)")
    save_figure(args.saving_dir, "generalization_error_top1")
    plt.close()

    # plot top 1 per imp iter
    colormap = plt.get_cmap("hot")
    colors = [colormap(i / len(list_epochs)) for i in range(len(list_epochs))]
    list_test_losses = np.split(test_losses, splitting_indices)
    list_train_losses = np.split(train_losses, splitting_indices)
    list_test_top1 = np.split(test_top1, splitting_indices)
    list_test_train_top1 = np.split(train_top1, splitting_indices)
    list_generalization_error_top1 = np.split(
        generalization_error_top1, splitting_indices
    )
    list_pathnorm1 = np.split(pathnorm1, splitting_indices)
    list_pathnorm4 = np.split(pathnorm4, splitting_indices)
    list_pathnorm2 = np.split(pathnorm2, splitting_indices)
    for i, (e, tr, te, g) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
        )
    ):
        # plt.plot(e, tr, label="Train Top 1")
        plt.plot(e[:-1], te[:-1], label=f"{i}", color=colors[i])
        # plt.plot(e, g, label="generalization error")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Test Top 1 Accuracy")
    save_figure(args.saving_dir, "test_top_1_per_iter")
    plt.close()

    # plot train and test losses per imp iter
    for i, (e, tr, te) in enumerate(
        zip(list_epochs, list_train_losses, list_test_losses)
    ):
        plt.plot(e, tr, label="Train loss", color=colors[i])
        plt.plot(e, te, label="Test loss", color=colors[i])

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    save_figure(args.saving_dir, "train_test_losses_per_imp_iter")
    plt.close()

    # plot top 1 generalization error per imp iter
    for i, (e, tr, te, g) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
        )
    ):
        # plt.plot(e, tr, label="Train Top 1")
        plt.plot(e[:-1], g[:-1], label=f"{i}", color=colors[i])
        # plt.plot(e, g, label="generalization error")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "top_1_generalization_error_per_iter")
    plt.close()

    # plot top 1 generalization error per imp iter
    for i, (e, tr, te, g) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
        )
    ):
        # plt.plot(e, tr, label="Train Top 1")
        plt.plot(e[:-1], tr[:-1], label=f"{i}", color=colors[i])
        # plt.plot(e, g, label="generalization error")

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Train Top 1 Accuracy")
    save_figure(args.saving_dir, "train_top_1_per_iter")
    plt.close()

    # scatter plot top 1 generalization error per imp iter vs Lp path-norm
    for i, (e, tr, te, g, p) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
            list_pathnorm1,
        )
    ):
        plt.scatter(p[:-1], g[:-1], label=f"{i}", color=colors[i])

    plt.legend()
    plt.xlabel("L1 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_L1")
    plt.close()

    # scatter plot top 1 generalization error per imp iter vs Lp path-norm
    for i, (e, tr, te, g, p) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
            list_pathnorm2,
        )
    ):
        plt.scatter(p[:-1], g[:-1], label=f"{i}", color=colors[i])

    plt.legend()
    plt.xlabel("L2 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_L2")
    plt.close()

    # scatter plot top 1 generalization error per imp iter vs Lp path-norm
    for i, (e, tr, te, g, p) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
            list_pathnorm4,
        )
    ):
        plt.scatter(p[:-1], g[:-1], label=f"{i}", color=colors[i])

    plt.legend()
    plt.xlabel("L4 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_L4")
    plt.close()

    # scatter plot log Lp path norm
    for i, (e, tr, te, g, p) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
            list_pathnorm1,
        )
    ):
        plt.scatter(np.log(p[:-1]), g[:-1], label=f"{i}", color=colors[i])

    plt.legend()
    plt.xlabel("Log L1 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_log_L1")
    plt.close()

    # scatter plot log Lp path norm
    for i, (e, tr, te, g, p) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
            list_pathnorm2,
        )
    ):
        plt.scatter(np.log(p[:-1]), g[:-1], label=f"{i}", color=colors[i])

    plt.legend()
    plt.xlabel("Log L2 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_log_L2")
    plt.close()

    # scatter plot log Lp path norm
    for i, (e, tr, te, g, p) in enumerate(
        zip(
            list_epochs,
            list_test_train_top1,
            list_test_top1,
            list_generalization_error_top1,
            list_pathnorm4,
        )
    ):
        plt.scatter(np.log(p[:-1]), g[:-1], label=f"{i}", color=colors[i])

    plt.legend()
    plt.xlabel("Log L4 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_log_L4")
    plt.close()

    # scatter plot with color depending on epoch
    colors_epochs = [colormap(e / epochs.max()) for e in epochs]
    for color, g, p in zip(colors_epochs, generalization_error_top1, pathnorm1):
        plt.scatter(p, g, color=color)

    plt.xlabel("L1 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_L1_color_epoch")
    plt.close()

    # scatter plot with color depending on epoch
    for color, g, p in zip(colors_epochs, generalization_error_top1, pathnorm2):
        plt.scatter(p, g, color=color)

    plt.xlabel("L2 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_L2_color_epoch")
    plt.close()

    # scatter plot with color depending on epoch
    for color, g, p in zip(colors_epochs, generalization_error_top1, pathnorm4):
        plt.scatter(p, g, color=color)

    plt.xlabel("L4 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_L4_color_epoch")
    plt.close()

    # scatter plot with color depending on epoch
    colors_epochs = [colormap(e / epochs.max()) for e in epochs]
    for color, g, p in zip(colors_epochs, generalization_error_top1, pathnorm1):
        plt.scatter(np.log(p), g, color=color)

    plt.xlabel("L1 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_log_L1_color_epoch")
    plt.close()

    # scatter plot with color depending on epoch
    for color, g, p in zip(colors_epochs, generalization_error_top1, pathnorm2):
        plt.scatter(np.log(p), g, color=color)

    plt.xlabel("L2 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_log_L2_color_epoch")
    plt.close()

    # scatter plot with color depending on epoch
    for color, g, p in zip(colors_epochs, generalization_error_top1, pathnorm4):
        plt.scatter(np.log(p), g, color=color)

    plt.xlabel("L4 path-norm")
    plt.ylabel("Top 1 Generalization Error")
    save_figure(args.saving_dir, "scatter_log_L4_color_epoch")
    plt.close()

    # plot path-norms
    plt.plot(epochs, pathnorm1, label="L1")
    plt.plot(epochs, pathnorm2, label="L2")
    plt.plot(epochs, pathnorm4, label="L4")
    plt.plot(epochs, pathnorm8, label="L8")
    plt.plot(epochs, pathnorm16, label="L16")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Path-norm")

    save_figure(args.saving_dir, "path_norm")
    plt.close()

    # plot L1 path-norm
    plt.plot(epochs, pathnorm1)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L1 path-norm")
    save_figure(args.saving_dir, "L1_path_norm")
    plt.close()

    # plot L1 path-norm per imp iter
    # split pathnorms to get the pathnorms of each imp iter
    for i, (e, p) in enumerate(zip(list_epochs, list_pathnorm1)):
        plt.plot(e, p, label=f"{i}", color=colors[i])
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L1 path-norm")
    save_figure(args.saving_dir, "L1_path_norm_per_iter")
    plt.close()

    # plot L2 path-norm per imp iter
    # split pathnorms to get the pathnorms of each imp iter
    for i, (e, p) in enumerate(zip(list_epochs, list_pathnorm2)):
        plt.plot(e, p, label=f"{i}", color=colors[i])
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L2 path-norm")
    save_figure(args.saving_dir, "L2_path_norm_per_iter")
    plt.close()

    # plot L4 path-norm per imp iter
    # split pathnorms to get the pathnorms of each imp iter
    for i, (e, p) in enumerate(zip(list_epochs, list_pathnorm4)):
        plt.plot(e, p, label=f"{i}", color=colors[i])
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("L4 path-norm")
    save_figure(args.saving_dir, "L4_path_norm_per_iter")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-training-dir",
        type=Path,
        help="path to the directory containing the results of the training",
    )
    parser.add_argument(
        "--saving-dir", type=Path, help="directory to save the figures"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=-1,
        help="node rank for distributed training",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=2, help="number of seeds"
    )
    args = parser.parse_args()

    main(args)
