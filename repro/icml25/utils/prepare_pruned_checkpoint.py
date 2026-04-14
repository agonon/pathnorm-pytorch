from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .pipeline import (
    RunMetadata,
    apply_rescaling_if_needed,
    apply_variant_pruning,
    best_checkpoint_path,
    build_resnet18,
    load_checkpoint,
    metadata_path,
    pruned_checkpoint_path,
    save_metadata,
    variant_is_rescaled,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a one-shot pruned checkpoint for the ICML 2025 reproduction."
    )
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        type=str,
        choices=[
            "path-magnitude",
            "path-magnitude-rescaled",
            "magnitude",
            "magnitude-rescaled",
        ],
        required=True,
    )
    parser.add_argument("--pruning-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = pruned_checkpoint_path(args.run_dir)
    if checkpoint_path.exists() and not args.overwrite:
        print(f"Pruned checkpoint already exists at {checkpoint_path}.")
        return

    dense_best_path = best_checkpoint_path(args.dense_dir)
    if not dense_best_path.exists():
        raise FileNotFoundError(
            f"Could not find the dense best checkpoint at `{dense_best_path}`."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense_checkpoint = load_checkpoint(dense_best_path, device)
    model = build_resnet18(device)
    model.load_state_dict(dense_checkpoint["state_dict"])
    model.eval()

    apply_rescaling_if_needed(model, variant=args.variant, seed=args.seed)
    apply_variant_pruning(
        model,
        variant=args.variant,
        pruning_ratio=args.pruning_ratio,
        device=device,
    )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(dense_checkpoint.get("epoch", 0)),
            "arch": dense_checkpoint.get("arch", "resnet18"),
            "state_dict": model.state_dict(),
            "best_val_top1": dense_checkpoint.get("best_val_top1", 0.0),
            "variant": args.variant,
            "pruning_ratio": args.pruning_ratio,
            "rescaled_before_pruning": variant_is_rescaled(args.variant),
        },
        checkpoint_path,
    )
    save_metadata(
        args.run_dir,
        RunMetadata(
            variant=args.variant,
            pruning_ratio=args.pruning_ratio,
            seed=args.seed,
            arch="resnet18",
            dense_dir=str(args.dense_dir),
            rescaled_before_pruning=variant_is_rescaled(args.variant),
        ),
    )
    print(f"Saved pruned checkpoint to {checkpoint_path}.")


if __name__ == "__main__":
    main()
