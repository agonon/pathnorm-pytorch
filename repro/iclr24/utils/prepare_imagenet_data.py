#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from repro.iclr24.utils.imagenet_data import check_imagenet_prepared
from repro.iclr24.utils.imagenet_data import stage_imagenet


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage an already-downloaded ImageNet directory for the reproduction notebooks. "
            "Because ImageNet is licensed, this helper does not fetch it from the internet."
        )
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Target directory that should contain train/ and val/ after staging.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Prepared ImageNet directory that already contains train/ and val/.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to stage the data into dest-dir.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check whether dest-dir is ready.",
    )
    args = parser.parse_args()

    dest_status = check_imagenet_prepared(args.dest_dir)
    if args.check_only:
        print(dest_status.message)
        return 0 if dest_status.prepared else 1

    if args.source_dir is None:
        print(dest_status.message)
        print(
            "No `--source-dir` was provided. Either materialize ImageNet first with "
            "`python -m repro.iclr24.utils.download_imagenet_from_hf`, or prepare it elsewhere and then "
            "rerun this helper with `--source-dir /path/to/prepared/imagenet`."
        )
        return 1

    status = stage_imagenet(args.source_dir, args.dest_dir, mode=args.mode)
    print(status.message)
    return 0 if status.prepared else 1


if __name__ == "__main__":
    raise SystemExit(main())
