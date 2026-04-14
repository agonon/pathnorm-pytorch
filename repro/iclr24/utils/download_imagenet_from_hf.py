#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import shutil

from datasets import Image as HFImage
from datasets import load_dataset
from datasets import load_dataset_builder
from PIL import Image as PILImage
from tqdm.auto import tqdm

from repro.iclr24.utils.imagenet_data import check_imagenet_prepared


HF_REPO_ID = "ILSVRC/imagenet-1k"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_VAL_SPLIT = "validation"

FORMAT_TO_SUFFIX = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download ImageNet-1k from the gated Hugging Face mirror and materialize "
            "a torchvision-compatible directory tree with train/ and val/ folders."
        )
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Target directory that will contain train/ and val/.",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Hugging Face dataset repo id (default: {HF_REPO_ID}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help=(
            "Hugging Face token for gated access. Defaults to HF_TOKEN if set. "
            "If omitted, the script falls back to the locally logged-in token."
        ),
    )
    parser.add_argument(
        "--train-split",
        default=DEFAULT_TRAIN_SPLIT,
        help=f"Name of the train split on Hugging Face (default: {DEFAULT_TRAIN_SPLIT}).",
    )
    parser.add_argument(
        "--val-split",
        default=DEFAULT_VAL_SPLIT,
        help=f"Name of the validation split on Hugging Face (default: {DEFAULT_VAL_SPLIT}).",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional cap used for smoke tests; when omitted, materialize the full split.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream directly from Hugging Face instead of caching the full dataset locally first.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already-materialized image files.",
    )
    return parser.parse_args()


def resolve_token(token: str | None) -> str | bool:
    return token if token else True


def infer_suffix(image_bytes: bytes) -> str:
    with PILImage.open(BytesIO(image_bytes)) as image:
        image_format = image.format or "JPEG"
    return FORMAT_TO_SUFFIX.get(image_format.upper(), ".jpg")


def class_dir_name(label: int) -> str:
    return f"{int(label):04d}"


def dest_path_for_example(
    split_dir: Path,
    split_index: int,
    label: int,
    image_payload: dict,
) -> Path:
    class_dir = split_dir / class_dir_name(label)
    class_dir.mkdir(parents=True, exist_ok=True)

    source_path = image_payload.get("path")
    if source_path:
        suffix = Path(source_path).suffix.lower() or ".jpg"
    elif image_payload.get("bytes") is not None:
        suffix = infer_suffix(image_payload["bytes"])
    else:
        raise ValueError("Image payload does not contain a path or bytes.")

    return class_dir / f"{split_index:08d}{suffix}"


def write_example_image(
    *,
    image_payload: dict,
    destination: Path,
    overwrite: bool,
) -> None:
    if destination.exists() and not overwrite:
        return

    source_path = image_payload.get("path")
    if source_path and Path(source_path).is_file():
        shutil.copyfile(source_path, destination)
        return

    image_bytes = image_payload.get("bytes")
    if image_bytes is None:
        raise ValueError(f"Could not materialize image for `{destination}`.")
    destination.write_bytes(image_bytes)


def materialize_split(
    *,
    repo_id: str,
    split_name: str,
    output_split_name: str,
    dest_dir: Path,
    token: str | bool,
    cache_dir: Path | None,
    limit_per_split: int | None,
    streaming: bool,
    overwrite: bool,
    total: int | None,
) -> dict[str, int | str]:
    dataset = load_dataset(
        repo_id,
        split=split_name,
        token=token,
        cache_dir=None if cache_dir is None else str(cache_dir),
        streaming=streaming,
    )
    dataset = dataset.cast_column("image", HFImage(decode=False))

    split_dir = dest_dir / output_split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    progress = tqdm(
        dataset,
        total=total if limit_per_split is None else min(total or limit_per_split, limit_per_split),
        desc=f"{output_split_name}",
        unit="img",
    )
    for split_index, example in enumerate(progress):
        if limit_per_split is not None and split_index >= limit_per_split:
            break

        label = int(example["label"])
        image_payload = example["image"]
        destination = dest_path_for_example(
            split_dir=split_dir,
            split_index=split_index,
            label=label,
            image_payload=image_payload,
        )
        write_example_image(
            image_payload=image_payload,
            destination=destination,
            overwrite=overwrite,
        )
        written += 1

    return {
        "split": output_split_name,
        "source_split": split_name,
        "written": written,
    }


def write_metadata(
    *,
    dest_dir: Path,
    repo_id: str,
    train_split: str,
    val_split: str,
    builder,
    streaming: bool,
) -> None:
    label_feature = builder.info.features["label"]
    payload = {
        "repo_id": repo_id,
        "train_split": train_split,
        "val_split": val_split,
        "streaming": streaming,
        "label_names": getattr(label_feature, "names", None),
    }
    (dest_dir / "hf_imagenet_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def main() -> int:
    args = parse_args()
    dest_dir = args.dest_dir.expanduser().resolve()
    cache_dir = None if args.cache_dir is None else args.cache_dir.expanduser().resolve()
    token = resolve_token(args.token)

    dest_dir.mkdir(parents=True, exist_ok=True)
    builder = load_dataset_builder(
        args.repo_id,
        token=token,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    write_metadata(
        dest_dir=dest_dir,
        repo_id=args.repo_id,
        train_split=args.train_split,
        val_split=args.val_split,
        builder=builder,
        streaming=args.streaming,
    )

    split_totals = {}
    if builder.info.splits is not None:
        split_totals["train"] = builder.info.splits[args.train_split].num_examples
        split_totals["val"] = builder.info.splits[args.val_split].num_examples

    train_summary = materialize_split(
        repo_id=args.repo_id,
        split_name=args.train_split,
        output_split_name="train",
        dest_dir=dest_dir,
        token=token,
        cache_dir=cache_dir,
        limit_per_split=args.limit_per_split,
        streaming=args.streaming,
        overwrite=args.overwrite,
        total=split_totals.get("train"),
    )
    val_summary = materialize_split(
        repo_id=args.repo_id,
        split_name=args.val_split,
        output_split_name="val",
        dest_dir=dest_dir,
        token=token,
        cache_dir=cache_dir,
        limit_per_split=args.limit_per_split,
        streaming=args.streaming,
        overwrite=args.overwrite,
        total=split_totals.get("val"),
    )

    summary_path = dest_dir / "hf_imagenet_download_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "streaming": args.streaming,
                "train": train_summary,
                "val": val_summary,
            },
            indent=2,
        )
        + "\n"
    )

    status = check_imagenet_prepared(dest_dir)
    print(status.message)
    print(f"Wrote metadata to `{dest_dir / 'hf_imagenet_metadata.json'}`.")
    print(f"Wrote summary to `{summary_path}`.")
    return 0 if status.prepared else 1


if __name__ == "__main__":
    raise SystemExit(main())
