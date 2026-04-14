#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from repro.iclr24.utils.imagenet_data import verify_imagenet_file_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a prepared ImageNet tree has the expected train/val file counts."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing train/ and val/ subdirectories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = verify_imagenet_file_counts(args.root)
    print(status.message)
    return 0 if status.prepared else 1


if __name__ == "__main__":
    raise SystemExit(main())
