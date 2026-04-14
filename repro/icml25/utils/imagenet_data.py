from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass
class ImageNetStatus:
    prepared: bool
    root: Path
    message: str


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
