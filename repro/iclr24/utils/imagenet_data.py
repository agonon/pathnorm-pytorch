from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

EXPECTED_IMAGENET_TRAIN_FILES = 1_281_167
EXPECTED_IMAGENET_VAL_FILES = 50_000


@dataclass
class ImageNetStatus:
    prepared: bool
    root: Path
    message: str


def count_files_recursive(root: str | Path) -> int:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return 0

    count = 0
    stack = [root]
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_file():
                    count += 1
                elif entry.is_dir():
                    stack.append(Path(entry.path))
    return count


def check_imagenet_prepared(root: str | Path) -> ImageNetStatus:
    root = Path(root).expanduser().resolve()
    train_dir = root / "train"
    val_dir = root / "val"

    if not root.exists():
        return ImageNetStatus(
            prepared=False,
            root=root,
            message=(
                f"ImageNet directory `{root}` does not exist. "
                "Prepare the dataset first, then stage it with the local reproduction helpers."
            ),
        )

    missing = [str(path.name) for path in (train_dir, val_dir) if not path.is_dir()]
    if missing:
        return ImageNetStatus(
            prepared=False,
            root=root,
            message=(
                f"ImageNet directory `{root}` is missing the expected subdirectories: {', '.join(missing)}."
            ),
        )

    if not any(train_dir.iterdir()) or not any(val_dir.iterdir()):
        return ImageNetStatus(
            prepared=False,
            root=root,
            message=(
                f"ImageNet directory `{root}` contains empty `train/` or `val/` folders."
            ),
        )

    return ImageNetStatus(
        prepared=True,
        root=root,
        message=f"ImageNet appears ready under `{root}`.",
    )


def stage_imagenet(
    source_dir: str | Path,
    dest_dir: str | Path,
    mode: str = "symlink",
) -> ImageNetStatus:
    source_dir = Path(source_dir).expanduser().resolve()
    dest_dir = Path(dest_dir).expanduser().resolve()

    source_status = check_imagenet_prepared(source_dir)
    if not source_status.prepared:
        raise FileNotFoundError(source_status.message)

    dest_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        source_split = source_dir / split
        dest_split = dest_dir / split

        if dest_split.exists():
            if dest_split.is_symlink() and dest_split.resolve() == source_split:
                continue
            if dest_split.is_dir() and any(dest_split.iterdir()):
                continue
            raise FileExistsError(
                f"Destination path `{dest_split}` already exists and cannot be replaced automatically."
            )

        if mode == "symlink":
            dest_split.symlink_to(source_split, target_is_directory=True)
        elif mode == "copy":
            shutil.copytree(source_split, dest_split)
        else:
            raise ValueError(f"Unsupported staging mode: {mode!r}")

    return check_imagenet_prepared(dest_dir)


def verify_imagenet_file_counts(root: str | Path) -> ImageNetStatus:
    base_status = check_imagenet_prepared(root)
    if not base_status.prepared:
        return base_status

    root = base_status.root
    train_count = count_files_recursive(root / "train")
    val_count = count_files_recursive(root / "val")

    if train_count != EXPECTED_IMAGENET_TRAIN_FILES or val_count != EXPECTED_IMAGENET_VAL_FILES:
        return ImageNetStatus(
            prepared=False,
            root=root,
            message=(
                f"ImageNet under `{root}` has incomplete file counts: "
                f"train={train_count}/{EXPECTED_IMAGENET_TRAIN_FILES}, "
                f"val={val_count}/{EXPECTED_IMAGENET_VAL_FILES}."
            ),
        )

    return ImageNetStatus(
        prepared=True,
        root=root,
        message=(
            f"ImageNet is complete under `{root}`: "
            f"train={train_count}, val={val_count}."
        ),
    )
