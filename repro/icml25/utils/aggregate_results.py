from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import (
    build_accuracy_tables,
    build_dense_summary_frame,
    build_training_curve_payload,
)
from .plot_training_curves import FIGURE_ARTIFACT_NAME, save_training_curves_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ICML 2025 pruning runs into paper-facing artifacts."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root results directory, typically repro/icml25/results/rerun.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)

    dense_df = build_dense_summary_frame(args.results_root)
    if dense_df.empty:
        print("No dense runs found yet for dense_summary.csv.")
    else:
        dense_path = args.results_root / "dense_summary.csv"
        dense_df.to_csv(dense_path, index=False)
        print(f"Wrote {dense_path}.")

    body_df, full_df = build_accuracy_tables(args.results_root)
    if body_df.empty:
        print("No pruning runs found yet for pruning_accuracy_body.csv.")
    else:
        body_path = args.results_root / "pruning_accuracy_body.csv"
        body_df.to_csv(body_path, index=False)
        print(f"Wrote {body_path}.")

    if full_df.empty:
        print("No pruning runs found yet for pruning_accuracy_full.csv.")
    else:
        full_path = args.results_root / "pruning_accuracy_full.csv"
        full_df.to_csv(full_path, index=False)
        print(f"Wrote {full_path}.")

    curve_payload = build_training_curve_payload(args.results_root)
    if curve_payload is None:
        print("No pruning runs found yet for training_curves_0.4.json.")
    else:
        curve_path = args.results_root / "training_curves_0.4.json"
        curve_path.write_text(json.dumps(curve_payload, indent=2) + "\n")
        print(f"Wrote {curve_path}.")
        figure_path = args.results_root / FIGURE_ARTIFACT_NAME
        try:
            save_training_curves_figure(curve_payload, figure_path)
            print(f"Wrote {figure_path}.")
        except Exception as exc:
            print(f"Could not write {figure_path}: {exc}")


if __name__ == "__main__":
    main()
