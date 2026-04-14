from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import torch
import torchvision.models as models

from pathnorm import compute_path_norms


DEFAULT_ARCHES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
DEFAULT_EXPONENTS = (1, 2, 4, 8, 16)
ARCH_TO_DEPTH = {arch: int(arch.replace("resnet", "")) for arch in DEFAULT_ARCHES}
DATASET_FRACTIONS = {
    39636: "1/32",
    79272: "1/16",
    158544: "1/8",
    317089: "1/4",
    634178: "1/2",
}
_EPOCHS_RE = re.compile(r"epochs=(\d+)")
_IMP_ITERS_RE = re.compile(r"imp_iters=(\d+)")
_SEED_RE = re.compile(r"seed=(\d+)")
_SIZE_DATASET_RE = re.compile(r"size_dataset=(\d+)")
_RANK_RE = re.compile(r"rank=([^/]+)")
FIGURE4_PAPER_PDFS = (
    "L1_path_norm_per_iter.pdf",
    "L2_path_norm_per_iter.pdf",
    "L4_path_norm_per_iter.pdf",
    "top_1_generalization_error_per_iter.pdf",
    "train_top_1_per_iter.pdf",
    "test_top_1_per_iter.pdf",
)
FIGURE5_PAPER_PDFS = (
    "L1_path_norm.pdf",
    "top1.pdf",
    "cross_entropy.pdf",
)


def get_resnet_builders():
    return {
        "resnet18": lambda: models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        ),
        "resnet34": lambda: models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1
        ),
        "resnet50": lambda: models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        ),
        "resnet101": lambda: models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2
        ),
        "resnet152": lambda: models.resnet152(
            weights=models.ResNet152_Weights.IMAGENET1K_V2
        ),
    }


def compute_pretrained_pathnorm_dataframe(
    *,
    results_dir: Path,
    device: torch.device,
    exponents: tuple[int, ...] = DEFAULT_EXPONENTS,
    input_shape: tuple[int, int, int] = (3, 224, 224),
) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for arch, build_model in get_resnet_builders().items():
        model = build_model().to(device).eval()
        values = compute_path_norms(
            model,
            input_shape=input_shape,
            exponents=exponents,
            device=device,
            working_dtype=torch.float64,
        )

        arch_dir = results_dir / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        tensor_values = torch.tensor(values, dtype=torch.float64)
        torch.save(tensor_values, arch_dir / "pathnorms.pt")

        for exponent, value in zip(exponents, values):
            log10_value = (
                math.log10(value)
                if math.isfinite(value) and value > 0
                else float("nan")
            )
            rows.append(
                {
                    "arch": arch,
                    "q": exponent,
                    "pathnorm": float(value),
                    "log10_pathnorm": float(log10_value),
                    "finite_pathnorm": bool(math.isfinite(value)),
                }
            )

    pretrained_df = pd.DataFrame(rows)
    pretrained_df["depth"] = pretrained_df["arch"].map(ARCH_TO_DEPTH)
    pretrained_df = pretrained_df.sort_values(["depth", "q"]).reset_index(drop=True)
    pretrained_df.to_csv(results_dir / "pathnorm_summary.csv", index=False)
    (results_dir / "pathnorm_metadata.json").write_text(
        json.dumps(
            {
                "working_dtype": "torch.float64",
                "exponents": list(exponents),
                "input_shape": list(input_shape),
                "pathnorm_api": "compute_path_norms",
            },
            indent=2,
        )
        + "\n"
    )
    return pretrained_df


def load_pretrained_pathnorm_dataframe(
    results_dir: Path,
    *,
    exponents: tuple[int, ...] = DEFAULT_EXPONENTS,
    arches: tuple[str, ...] = DEFAULT_ARCHES,
) -> pd.DataFrame:
    summary_csv = results_dir / "pathnorm_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
    else:
        rows: list[dict[str, float | int | str]] = []
        for arch in arches:
            tensor_path = results_dir / arch / "pathnorms.pt"
            if not tensor_path.exists():
                continue
            values = torch.load(tensor_path, map_location="cpu").double()
            for exponent, value in zip(exponents, values):
                float_value = float(value)
                rows.append(
                    {
                        "arch": arch,
                        "q": exponent,
                        "pathnorm": float_value,
                        "log10_pathnorm": (
                            math.log10(float_value)
                            if math.isfinite(float_value) and float_value > 0
                            else float("nan")
                        ),
                        "finite_pathnorm": bool(math.isfinite(float_value)),
                    }
                )
        df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["depth"] = df["arch"].map(ARCH_TO_DEPTH)
    return df.sort_values(["depth", "q"]).reset_index(drop=True)


def summarize_margin_quantiles(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for arch in DEFAULT_ARCHES:
        margins_path = results_root / arch / "margins.pt"
        if not margins_path.exists():
            continue

        margins = torch.load(margins_path, map_location="cpu").flatten().double()
        number_negative = int((margins < 0).sum().item())
        sorted_margins = torch.sort(margins).values
        train_top1 = 1.0 - number_negative / len(margins)
        quantile_indices = [
            number_negative,
            int((2 / 3) * number_negative + (len(margins) - 1) / 3),
            int(0.5 * number_negative + 0.5 * (len(margins) - 1)),
            int((1 / 3) * number_negative + 2 * (len(margins) - 1) / 3),
            len(margins) - 1,
        ]
        rows.append(
            {
                "arch": arch,
                "train_top1": train_top1,
                "margin_q=e": float(sorted_margins[quantile_indices[0]]),
                "margin_q=2e/3+1/3": float(sorted_margins[quantile_indices[1]]),
                "margin_q=(e+1)/2": float(sorted_margins[quantile_indices[2]]),
                "margin_q=e/3+2/3": float(sorted_margins[quantile_indices[3]]),
                "margin_q=1": float(sorted_margins[quantile_indices[4]]),
                "margin_min": float(sorted_margins[0]),
                "margin_median": float(sorted_margins[len(sorted_margins) // 2]),
                "margin_max": float(sorted_margins[-1]),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["depth"] = df["arch"].map(ARCH_TO_DEPTH)
    return df.sort_values("depth").drop(columns=["depth"]).reset_index(drop=True)


def compute_table2_dataframe(
    *,
    B_value: float = 2.640000104904175,
    train_fraction: float = 0.99,
) -> pd.DataFrame:
    n = int(train_fraction * 1281167)
    din = 224 * 224 * 3
    dout = 1000
    pooling_types = 1
    max_kernel = 9
    rows: list[dict[str, float | int]] = []

    for depth in [18, 34, 50, 101, 152]:
        term_1 = depth * math.log((3 + 2 * pooling_types) * max_kernel)
        frac = (3 + 2 * pooling_types) / (1 + pooling_types)
        term_2 = math.log(frac * (din + 1) * dout)
        c_value = math.sqrt(term_1 + term_2)
        bound = c_value * B_value * 4 / math.sqrt(n)

        maxpool_count = 1
        sharpened_term_1 = depth * math.log(3) + maxpool_count * math.log(max_kernel)
        sharpened_term_2 = math.log((din + 1) * dout)
        c_sharp = math.sqrt(sharpened_term_1 + sharpened_term_2)
        sharpened_bound = c_sharp * B_value * 4 / math.sqrt(n)

        rows.append(
            {
                "arch": f"resnet{depth}",
                "depth": depth,
                "bound": bound,
                "sharpened_bound": sharpened_bound,
            }
        )

    return pd.DataFrame(rows).sort_values("depth").reset_index(drop=True)


def load_imp_training_dataframe(results_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_csv_path)
    df = df.dropna(subset=["batch_size"]).copy()
    numeric_columns = [
        "epoch",
        "train/loss",
        "test/loss",
        "train/acc1",
        "test/acc1",
        "pathnorm1",
        "pathnorm2",
        "pathnorm4",
        "pathnorm8",
        "pathnorm16",
        "imp_iter",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = (
        df.sort_values(["imp_iter", "epoch"])
        .drop_duplicates(subset=["imp_iter", "epoch"], keep="first")
        .reset_index(drop=True)
    )
    df["generalization_error_top1"] = df["train/acc1"] - df["test/acc1"]
    df["generalization_error_ce"] = df["test/loss"] - df["train/loss"]
    return df


def _rank_preference_key(results_csv_path: Path) -> tuple[int, str]:
    rank_match = _RANK_RE.search(str(results_csv_path))
    rank = rank_match.group(1) if rank_match else ""
    if rank == "0":
        return (0, rank)
    if rank == "-1":
        return (1, rank)
    return (2, rank)


def find_imp_results_csv(
    results_root: Path,
    *,
    seed: int = 0,
    arch: str = "resnet18",
) -> Path | None:
    search_root = Path(results_root) / "2_train_imp" / f"seed={seed}" / arch
    if not search_root.exists():
        return None

    candidates = sorted(
        search_root.glob("*/rank=*/csv/results.csv"),
        key=lambda path: (
            _rank_preference_key(path),
            str(path),
        ),
    )
    if not candidates:
        return None
    return candidates[0]


def describe_imp_results_csv(results_csv_path: Path) -> dict[str, int | str]:
    run_dir = results_csv_path.parents[2]
    run_name = run_dir.name
    epochs_match = _EPOCHS_RE.search(run_name)
    imp_iters_match = _IMP_ITERS_RE.search(run_name)
    rank_match = _RANK_RE.search(str(results_csv_path))
    return {
        "run_dir": str(run_dir),
        "epochs": int(epochs_match.group(1)) if epochs_match else -1,
        "imp_iters": int(imp_iters_match.group(1)) if imp_iters_match else -1,
        "rank": rank_match.group(1) if rank_match else "",
    }


def _find_saved_plot_dir(
    results_root: Path,
    *,
    stage_dir: str,
    arch: str,
    anchor_pdf: str,
) -> Path | None:
    search_root = Path(results_root) / stage_dir
    if not search_root.exists():
        return None

    candidates = sorted(
        path
        for path in search_root.glob(f"num_seeds=*/{arch}/lr=*")
        if path.is_dir() and (path / anchor_pdf).exists()
    )
    if not candidates:
        return None
    return candidates[0]


def find_figure4_plot_dir(
    results_root: Path,
    *,
    arch: str = "resnet18",
) -> Path | None:
    return _find_saved_plot_dir(
        results_root,
        stage_dir="3_plot_imp",
        arch=arch,
        anchor_pdf="test_top_1_per_iter.pdf",
    )


def find_figure5_plot_dir(
    results_root: Path,
    *,
    arch: str = "resnet18",
) -> Path | None:
    return _find_saved_plot_dir(
        results_root,
        stage_dir="5_plot_increasing_dataset",
        arch=arch,
        anchor_pdf="top1.pdf",
    )


def summarize_imp_final_epochs(imp_df: pd.DataFrame) -> pd.DataFrame:
    return (
        imp_df.sort_values(["imp_iter", "epoch"])
        .groupby("imp_iter", as_index=False)
        .tail(1)
        .loc[
            :,
            [
                "imp_iter",
                "epoch",
                "test/acc1",
                "train/acc1",
                "generalization_error_top1",
                "pathnorm1",
                "pathnorm2",
                "pathnorm4",
            ],
        ]
        .reset_index(drop=True)
    )


def load_increasing_dataset_dataframe(
    results_root: Path,
    *,
    dataset_sizes: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    manifest = describe_increasing_dataset_results_root(results_root)
    available_dataset_sizes = (
        tuple(manifest["dataset_sizes"])
        if dataset_sizes is None
        else dataset_sizes
    )
    rows: list[pd.DataFrame] = []
    for size_dataset in available_dataset_sizes:
        per_seed: list[pd.DataFrame] = []
        matching_records = [
            record
            for record in manifest["records"]
            if record["size_dataset"] == size_dataset
        ]
        for record in matching_records:
            csv_path = Path(record["csv_path"])
            df = pd.read_csv(csv_path).copy()
            numeric_columns = [
                "epoch",
                "train/loss",
                "test/loss",
                "train/acc1",
                "test/acc1",
                "pathnorm1",
                "pathnorm2",
                "pathnorm4",
            ]
            for column in numeric_columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            df = (
                df.sort_values("epoch")
                .drop_duplicates(subset=["epoch"], keep="first")
                .reset_index(drop=True)
            )
            per_seed.append(df.loc[:, numeric_columns])

        if not per_seed:
            continue

        averaged = (
            pd.concat(per_seed, ignore_index=True)
            .groupby("epoch", as_index=False)
            .mean(numeric_only=True)
        )
        averaged["size_dataset"] = size_dataset
        averaged["size_fraction"] = DATASET_FRACTIONS[size_dataset]
        averaged["generalization_error_top1"] = (
            averaged["train/acc1"] - averaged["test/acc1"]
        )
        averaged["generalization_error_ce"] = (
            averaged["test/loss"] - averaged["train/loss"]
        )
        rows.append(averaged)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_increasing_dataset_final_epoch(
    increasing_df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        increasing_df.sort_values(["size_dataset", "epoch"])
        .groupby("size_dataset", as_index=False)
        .tail(1)
        .loc[
            :,
            [
                "size_dataset",
                "size_fraction",
                "epoch",
                "test/acc1",
                "train/acc1",
                "generalization_error_top1",
                "generalization_error_ce",
                "pathnorm1",
            ],
        ]
        .sort_values("size_dataset")
        .reset_index(drop=True)
    )


def describe_increasing_dataset_results_root(
    results_root: Path,
    *,
    arch: str = "resnet18",
) -> dict[str, object]:
    search_root = Path(results_root) / "4_train_increasing_dataset"
    candidates = sorted(search_root.glob(f"seed=*/{arch}/size_dataset=*/lr=*/rank=*/csv/results.csv"))
    by_key: dict[tuple[int, int], dict[str, object]] = {}
    for csv_path in candidates:
        csv_str = str(csv_path)
        seed_match = _SEED_RE.search(csv_str)
        size_dataset_match = _SIZE_DATASET_RE.search(csv_str)
        epochs_match = _EPOCHS_RE.search(csv_str)
        if seed_match is None or size_dataset_match is None or epochs_match is None:
            continue
        seed = int(seed_match.group(1))
        size_dataset = int(size_dataset_match.group(1))
        epochs = int(epochs_match.group(1))
        record = {
            "seed": seed,
            "size_dataset": size_dataset,
            "epochs": epochs,
            "csv_path": str(csv_path),
            "rank_preference": _rank_preference_key(csv_path),
        }
        key = (seed, size_dataset)
        previous = by_key.get(key)
        if previous is None or (
            record["rank_preference"],
            record["csv_path"],
        ) < (
            previous["rank_preference"],
            previous["csv_path"],
        ):
            by_key[key] = record

    records = sorted(
        by_key.values(),
        key=lambda record: (
            int(record["size_dataset"]),
            int(record["seed"]),
        ),
    )
    dataset_sizes = sorted({int(record["size_dataset"]) for record in records})
    seeds = sorted({int(record["seed"]) for record in records})
    epochs = sorted({int(record["epochs"]) for record in records})
    return {
        "records": records,
        "dataset_sizes": dataset_sizes,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "num_epochs": epochs[0] if len(epochs) == 1 else None,
    }
