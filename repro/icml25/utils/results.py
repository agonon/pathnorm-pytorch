from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .plot_training_curves import FIGURE_ARTIFACT_NAME


def paper_release_root(results_root: Path) -> Path:
    return Path(results_root) / "paper_release"


def rerun_root(results_root: Path) -> Path:
    return Path(results_root) / "rerun"


def artifact_path(
    results_root: Path,
    artifact_name: str,
    *,
    source: str = "auto",
) -> Path:
    results_root = Path(results_root)
    rerun_path = rerun_root(results_root) / artifact_name
    release_path = paper_release_root(results_root) / artifact_name

    if source == "auto":
        return rerun_path if rerun_path.exists() else release_path
    if source == "paper_release":
        return release_path
    if source == "rerun":
        return rerun_path
    raise ValueError(f"Unknown artifact source: {source}")


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json_if_exists(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def load_runtime_summary(results_root: Path, *, source: str = "auto") -> pd.DataFrame:
    return load_csv_if_exists(
        artifact_path(results_root, "benchmark_summary.csv", source=source)
    )


def load_dense_summary(results_root: Path, *, source: str = "auto") -> pd.DataFrame:
    return load_csv_if_exists(
        artifact_path(results_root, "dense_summary.csv", source=source)
    )


def load_body_accuracy_table(
    results_root: Path,
    *,
    source: str = "auto",
) -> pd.DataFrame:
    return load_csv_if_exists(
        artifact_path(results_root, "pruning_accuracy_body.csv", source=source)
    )


def load_full_accuracy_table(
    results_root: Path,
    *,
    source: str = "auto",
) -> pd.DataFrame:
    return load_csv_if_exists(
        artifact_path(results_root, "pruning_accuracy_full.csv", source=source)
    )


def load_training_curves(
    results_root: Path,
    *,
    source: str = "auto",
) -> dict | list | None:
    return load_json_if_exists(
        artifact_path(results_root, "training_curves_0.4.json", source=source)
    )


def load_training_curve_figure_path(
    results_root: Path,
    *,
    source: str = "auto",
) -> Path | None:
    path = artifact_path(results_root, FIGURE_ARTIFACT_NAME, source=source)
    return path if path.exists() else None


def _round_numeric_columns(df: pd.DataFrame, *, digits: int = 1) -> pd.DataFrame:
    formatted = df.copy()
    numeric_columns = formatted.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        formatted[column] = formatted[column].round(digits)
    return formatted


def format_runtime_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = ["network", "forward", "mag", "obd", "path_mag"]
    available = [column for column in columns if column in df.columns]
    formatted = df.loc[:, available].copy()
    return formatted.rename(
        columns={
            "network": "Network",
            "forward": "Forward",
            "mag": "Magnitude",
            "obd": "OBD",
            "path_mag": "Path-Mag",
        }
    )


def format_dense_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    columns = [
        column
        for column in ["row", "original_accuracy", "mean_val_acc1", "mean_best_epoch", "num_runs"]
        if column in df.columns
    ]
    formatted = _round_numeric_columns(df.loc[:, columns])
    return formatted.rename(
        columns={
            "row": "Setting",
            "original_accuracy": "Top-1 accuracy",
            "mean_val_acc1": "Mean val top-1",
            "mean_best_epoch": "Mean best epoch",
            "num_runs": "Runs",
        }
    )


def format_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    formatted = _round_numeric_columns(df)
    first_column = formatted.columns[0]
    return formatted.rename(columns={first_column: "Pruning method"})
