from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

from repro.iclr24.utils.imagenet_data import check_imagenet_prepared


def _repo_relative_path(path: Path) -> str:
    parts = path.as_posix().split("/")
    if "repro" in parts:
        return "/".join(parts[parts.index("repro") :])
    return path.as_posix()


def _imagenet_env_reference() -> str:
    return '"${IMAGENET_DIR:-data/imagenet}"'


def _hf_cache_env_reference() -> str:
    return '"${HF_IMAGENET_CACHE_DIR:-.hf-cache/imagenet}"'


def _env_command_block(env_lines: list[str], script_path: str) -> str:
    command_lines = [f"{env_lines[0]} \\"]
    command_lines.extend(f"{line} \\" for line in env_lines[1:])
    command_lines.append(script_path)
    return "\n".join(command_lines)


def run_command(
    command: list[str],
    *,
    repo_root: Path,
    env: dict[str, object] | None = None,
    cwd: Path | None = None,
) -> None:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update({key: str(value) for key, value in env.items()})
    effective_cwd = repo_root if cwd is None else cwd
    print("$", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=effective_cwd,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {' '.join(command)}"
        )


def imagenet_prepare_command(imagenet_dir: Path) -> str:
    return textwrap.dedent(
        """
        bash repro/iclr24/utils/prepare_imagenet.sh \
            --source-dir /path/to/already-prepared/imagenet \
            --dest-dir "${IMAGENET_DIR:-data/imagenet}"
        """
    ).strip()


def imagenet_download_from_hf_command(
    imagenet_dir: Path,
    hf_cache_dir: Path,
) -> str:
    return textwrap.dedent(
        """
        HF_TOKEN=... bash repro/iclr24/utils/download_imagenet_from_hf.sh \
            --streaming \
            --dest-dir "${IMAGENET_DIR:-data/imagenet}" \
            --cache-dir "${HF_IMAGENET_CACHE_DIR:-.hf-cache/imagenet}"
        """
    ).strip()


def require_imagenet(
    imagenet_dir: Path,
    *,
    hf_cache_dir: Path,
) -> None:
    status = check_imagenet_prepared(imagenet_dir)
    if status.prepared:
        print("ImageNet is ready.")
    if not status.prepared:
        raise RuntimeError(
            "ImageNet is not ready."
            + "\n\nOption 1 (download from Hugging Face):\n"
            + imagenet_download_from_hf_command(imagenet_dir, hf_cache_dir)
            + "\n\nOption 2 (stage an existing prepared copy):\n"
            + imagenet_prepare_command(imagenet_dir)
        )


def iclr24_wrapper_script(repo_root: Path, script_name: str) -> Path:
    return repo_root / "repro" / "iclr24" / "utils" / script_name


def _figure4_rerun_command(
    imagenet_dir: Path,
    results_root: Path,
    *,
    epochs: int,
    imp_iters: int,
    batch_size: int,
) -> str:
    training_dir = (
        results_root
        / "2_train_imp"
        / "seed=0"
        / "resnet18"
        / f"lr=0.1_wd=0.0001_epochs={epochs}_scheduler=multi-step_percentage_pruning=0.2_imp_iters={imp_iters}"
    )
    plot_dir = (
        results_root
        / "3_plot_imp"
        / "num_seeds=1"
        / "resnet18"
        / f"lr=0.1_wd=0.0001_epochs={epochs}_scheduler=multi-step_percentage_pruning=0.2_imp_iters={imp_iters}"
    )
    return textwrap.dedent(
        _env_command_block(
            [
                f"DATA_DIR={_imagenet_env_reference()}",
                f'SAVING_DIR="{_repo_relative_path(training_dir)}"',
                "ARCH=resnet18",
                "WORKERS=16",
                f"BATCH_SIZE={batch_size}",
                f"EPOCHS={epochs}",
                f"IMP_ITERS={imp_iters}",
                "PERCENTAGE_PRUNING=0.2",
                "SEED=0",
                "LR=0.1",
                "WD=0.0001",
                "LR_SCHEDULER=multi-step",
            ],
            "bash repro/iclr24/utils/run_figure4_train.sh",
        )
        + "\n\n"
        + _env_command_block(
            [
                f'RESULTS_TRAINING_DIR="{_repo_relative_path(training_dir)}"',
                f'SAVING_DIR="{_repo_relative_path(plot_dir)}"',
                "NUM_SEEDS=1",
                "RANK=-1",
                "ARCH=resnet18",
                f"EPOCHS={epochs}",
                f"IMP_ITERS={imp_iters}",
                "PERCENTAGE_PRUNING=0.2",
                "LR=0.1",
                "WD=0.0001",
                "LR_SCHEDULER=multi-step",
            ],
            "bash repro/iclr24/utils/run_figure4_plot.sh",
        )
    ).strip()


def figure4_rerun_command(
    imagenet_dir: Path,
    results_root: Path,
    *,
    smoke: bool = False,
) -> str:
    if smoke:
        return _figure4_rerun_command(
            imagenet_dir,
            results_root / "smoke",
            epochs=1,
            imp_iters=1,
            batch_size=512,
        )
    return _figure4_rerun_command(
        imagenet_dir,
        results_root,
        epochs=90,
        imp_iters=20,
        batch_size=1024,
    )


def figure4_qualitative_rerun_command(
    imagenet_dir: Path,
    results_root: Path,
) -> str:
    return _figure4_rerun_command(
        imagenet_dir,
        results_root / "qualitative",
        epochs=25,
        imp_iters=6,
        batch_size=1024,
    )


def figure4_downscaled_rerun_command(
    imagenet_dir: Path,
    results_root: Path,
) -> str:
    return _figure4_rerun_command(
        imagenet_dir,
        results_root / "mid_scale",
        epochs=40,
        imp_iters=8,
        batch_size=1024,
    )


def _figure5_rerun_command(
    imagenet_dir: Path,
    results_root: Path,
    *,
    epochs: int,
    batch_size: int,
    seeds: str,
    size_datasets: str,
    num_seeds: int,
) -> str:
    training_root = results_root / "4_train_increasing_dataset"
    plot_input_root = (
        training_root
        / "seed=0"
        / "resnet18"
        / ("size_dataset=39636")
        / f"lr=0.1_wd=0.0001_epochs={epochs}_scheduler=multi-step_percentage_pruning=_imp_iters=0"
    )
    plot_dir = (
        results_root
        / "5_plot_increasing_dataset"
        / f"num_seeds={num_seeds}"
        / "resnet18"
        / f"lr=0.1_wd=0.0001_epochs={epochs}_scheduler=multi-step_percentage_pruning=_imp_iters=0"
    )
    command = _env_command_block(
        [
            f"DATA_DIR={_imagenet_env_reference()}",
            f'SAVING_ROOT="{_repo_relative_path(training_root)}"',
            "ARCH=resnet18",
            "WORKERS=16",
            f"BATCH_SIZE={batch_size}",
            f"EPOCHS={epochs}",
            "IMP_ITERS=0",
            f'SEEDS="{seeds}"',
            f'SIZE_DATASETS="{size_datasets}"',
            "LR=0.1",
            "WD=0.0001",
            "LR_SCHEDULER=multi-step",
            "USE_TENSORBOARD=0",
        ],
        "bash repro/iclr24/utils/run_figure5_train.sh",
    )
    return (
        command
        + "\n"
        + "\n"
        + _env_command_block(
            [
                f'RESULTS_TRAINING_DIR="{_repo_relative_path(plot_input_root)}"',
                f'SAVING_DIR="{_repo_relative_path(plot_dir)}"',
                f"NUM_EPOCHS={epochs}",
                f"NUM_SEEDS={num_seeds}",
                "RANK=0",
                "ARCH=resnet18",
                "LR=0.1",
                "WD=0.0001",
                "LR_SCHEDULER=multi-step",
            ],
            "bash repro/iclr24/utils/run_figure5_plot.sh",
        )
    )


def figure5_rerun_command(
    imagenet_dir: Path,
    results_root: Path,
    *,
    smoke: bool = False,
) -> str:
    if smoke:
        return _figure5_rerun_command(
            imagenet_dir,
            results_root / "smoke",
            epochs=1,
            batch_size=512,
            seeds="0",
            size_datasets="39636 79272",
            num_seeds=1,
        )
    return _figure5_rerun_command(
        imagenet_dir,
        results_root,
        epochs=90,
        batch_size=1024,
        seeds="0 1 2",
        size_datasets="39636 79272 158544 317089 634178",
        num_seeds=3,
    )


def figure5_qualitative_rerun_command(
    imagenet_dir: Path,
    results_root: Path,
) -> str:
    return _figure5_rerun_command(
        imagenet_dir,
        results_root / "qualitative",
        epochs=25,
        batch_size=1024,
        seeds="0 1",
        size_datasets="39636 79272 158544 317089 634178",
        num_seeds=2,
    )


def figure5_downscaled_rerun_command(
    imagenet_dir: Path,
    results_root: Path,
) -> str:
    return _figure5_rerun_command(
        imagenet_dir,
        results_root / "mid_scale",
        epochs=90,
        batch_size=512,
        seeds="0",
        size_datasets="39636 79272 158544 317089 634178",
        num_seeds=1,
    )
