from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


FIGURE_ARTIFACT_NAME = "plot_training_curves_test0.4.pdf"
PREVIEW_ARTIFACT_NAME = "plot_training_curves_test0.4.png"
DISPLAY_ORDER = (
    "magnitude",
    "magnitude-rescaled",
    "path-magnitude",
    "path-magnitude-rescaled",
)
DISPLAY_STYLE = {
    "magnitude": {
        "label": "Magnitude w/o Rescale",
        "color": "red",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    "magnitude-rescaled": {
        "label": "Magnitude w/ Rescale",
        "color": "red",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "path-magnitude": {
        "label": "Path-Magnitude",
        "color": "blue",
        "linestyle": "-",
        "linewidth": 1.5,
    },
    # The paper states that this curve overlaps with path-magnitude after rescaling.
    "path-magnitude-rescaled": {
        "label": None,
        "color": "blue",
        "linestyle": "-",
        "linewidth": 1.5,
    },
}
LABEL_TO_VARIANT = {
    "Magnitude": "magnitude",
    "Magnitude (rescaled)": "magnitude-rescaled",
    "Path-magnitude": "path-magnitude",
    "Path-magnitude (rescaled)": "path-magnitude-rescaled",
}


def normalize_curve_payload(curve_payload: dict[str, object]) -> tuple[float | None, dict[str, dict[str, list[float]]]]:
    pruning_ratio = curve_payload.get("pruning_ratio")
    normalized: dict[str, dict[str, list[float]]] = {}

    for series in curve_payload.get("series", []):
        if not isinstance(series, dict):
            continue
        variant = series.get("variant")
        if not isinstance(variant, str):
            label = series.get("label")
            variant = LABEL_TO_VARIANT.get(label)
        if variant not in DISPLAY_STYLE:
            continue

        epochs = series.get("epoch", series.get("x", []))
        test_acc1 = series.get("test_acc1", series.get("top1", series.get("y", [])))
        if not isinstance(epochs, list) or not isinstance(test_acc1, list):
            continue
        if len(epochs) != len(test_acc1) or not epochs:
            continue

        dedup_by_epoch: dict[int, float] = {}
        for epoch, value in zip(epochs, test_acc1):
            try:
                dedup_by_epoch[int(epoch)] = float(value)
            except (TypeError, ValueError):
                continue
        if not dedup_by_epoch:
            continue

        ordered_epochs = sorted(dedup_by_epoch)
        normalized[variant] = {
            "epoch": ordered_epochs,
            "test_acc1": [dedup_by_epoch[epoch] for epoch in ordered_epochs],
        }

    return pruning_ratio if isinstance(pruning_ratio, (int, float)) else None, normalized


def plot_training_curves(curve_payload: dict[str, object]) -> tuple[plt.Figure, plt.Axes]:
    pruning_ratio, normalized = normalize_curve_payload(curve_payload)
    fig, ax = plt.subplots(figsize=(7, 4))

    plotted = 0
    for variant in DISPLAY_ORDER:
        series = normalized.get(variant)
        if series is None:
            continue
        style = DISPLAY_STYLE[variant]
        ax.plot(
            series["epoch"],
            series["test_acc1"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            label=style["label"],
        )
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        raise ValueError("No recognizable training-curve series found in payload.")

    max_epoch = max(max(series["epoch"]) for series in normalized.values())
    x_max = 90 if max_epoch <= 90 else max_epoch
    pruning_percent = int(round(100 * pruning_ratio)) if pruning_ratio is not None else 40
    ax.set_title(f"Pruning {pruning_percent}%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test top-1 accuracy")
    ax.set_xlim(0, x_max)
    if x_max == 90:
        ax.set_xticks([0, 20, 40, 60, 80])
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig, ax


def save_training_curves_figure(
    curve_payload: dict[str, object],
    saving_path: Path,
) -> Path:
    saving_path = Path(saving_path)
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_training_curves(curve_payload)
    try:
        fig.savefig(saving_path, bbox_inches="tight")
        if saving_path.suffix.lower() == ".pdf":
            fig.savefig(saving_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    finally:
        plt.close(fig)
    return saving_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the ICML 2025 pruning-curve figure in paper style."
    )
    parser.add_argument(
        "--payload",
        type=Path,
        required=True,
        help="Path to training_curves_0.4.json.",
    )
    parser.add_argument(
        "--saving-path",
        type=Path,
        required=True,
        help=f"Destination PDF path, for example {FIGURE_ARTIFACT_NAME}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    curve_payload = json.loads(args.payload.read_text())
    save_training_curves_figure(curve_payload, args.saving_path)


if __name__ == "__main__":
    main()
