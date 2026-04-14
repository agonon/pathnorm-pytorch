from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

from .imagenet_data import check_imagenet_prepared


def _repo_relative_path(path: Path) -> str:
    parts = Path(path).as_posix().split("/")
    if "repro" in parts:
        return "/".join(parts[parts.index("repro") :])
    return Path(path).as_posix()


def _imagenet_env_reference() -> str:
    return '"${IMAGENET_DIR:-data/imagenet}"'


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


def require_imagenet(imagenet_dir: Path) -> None:
    status = check_imagenet_prepared(imagenet_dir)
    print(status.message)
    if not status.prepared:
        raise RuntimeError(status.message)


def icml25_wrapper_script(repo_root: Path, script_name: str) -> Path:
    return repo_root / "repro" / "icml25" / "utils" / script_name


def _shell_join_ints(values: tuple[int, ...]) -> str:
    return " ".join(str(value) for value in values)


def runtime_benchmark_command(*, results_root: Path | None = None) -> str:
    env_lines: list[str] = []
    if results_root is not None:
        env_lines.append(f'RESULTS_ROOT="{_repo_relative_path(results_root)}"')
    if not env_lines:
        return "bash repro/icml25/utils/run_runtime_benchmark.sh"
    return _env_command_block(env_lines, "bash repro/icml25/utils/run_runtime_benchmark.sh")


def dense_train_command(
    data_dir: Path,
    *,
    seeds: tuple[int, ...],
    results_root: Path | None = None,
) -> str:
    env_lines = [f"DATA_DIR={_imagenet_env_reference()}"]
    if results_root is not None:
        env_lines.append(f'RESULTS_ROOT="{_repo_relative_path(results_root)}"')
    env_lines.append(f'SEEDS="{_shell_join_ints(seeds)}"')
    return textwrap.dedent(
        _env_command_block(env_lines, "bash repro/icml25/utils/run_dense_train.sh")
    ).strip()


def accuracy_sweep_command(
    data_dir: Path,
    *,
    seeds: tuple[int, ...],
    results_root: Path | None = None,
) -> str:
    env_lines = [f"DATA_DIR={_imagenet_env_reference()}"]
    if results_root is not None:
        env_lines.append(f'RESULTS_ROOT="{_repo_relative_path(results_root)}"')
    env_lines.append(f'SEEDS="{_shell_join_ints(seeds)}"')
    return textwrap.dedent(
        _env_command_block(env_lines, "bash repro/icml25/utils/run_accuracy_sweep.sh")
    ).strip()


def body_accuracy_command(
    data_dir: Path,
    *,
    seeds: tuple[int, ...],
    results_root: Path | None = None,
) -> str:
    env_lines = [f"DATA_DIR={_imagenet_env_reference()}"]
    if results_root is not None:
        env_lines.append(f'RESULTS_ROOT="{_repo_relative_path(results_root)}"')
    env_lines.extend(
        [
            f'SEEDS="{_shell_join_ints(seeds)}"',
            'VARIANTS="path-magnitude magnitude magnitude-rescaled"',
            'PRUNING_RATIOS="0.4 0.6 0.8"',
        ]
    )
    return textwrap.dedent(
        _env_command_block(env_lines, "bash repro/icml25/utils/run_accuracy_sweep.sh")
    ).strip()


def curve_repro_command(
    data_dir: Path,
    *,
    seeds: tuple[int, ...],
    results_root: Path | None = None,
) -> str:
    env_lines = [f"DATA_DIR={_imagenet_env_reference()}"]
    if results_root is not None:
        env_lines.append(f'RESULTS_ROOT="{_repo_relative_path(results_root)}"')
    env_lines.extend(
        [
            f'DENSE_SEEDS="{_shell_join_ints(seeds)}"',
            f'PRUNING_SEEDS="{_shell_join_ints(seeds)}"',
            'PRUNING_VARIANTS="path-magnitude-rescaled"',
            'PRUNING_RATIOS="0.4"',
            "PRECHECK_SMOKE=0",
        ]
    )
    return textwrap.dedent(
        _env_command_block(env_lines, "bash repro/icml25/utils/run_campaign.sh")
    ).strip()


def release_target_command(
    data_dir: Path,
    *,
    seeds: tuple[int, ...],
    results_root: Path | None = None,
) -> str:
    shared_env_lines = [f"DATA_DIR={_imagenet_env_reference()}"]
    if results_root is not None:
        shared_env_lines.append(f'RESULTS_ROOT="{_repo_relative_path(results_root)}"')
    shared_env_lines.extend(
        [
            f'DENSE_SEEDS="{_shell_join_ints(seeds)}"',
            f'PRUNING_SEEDS="{_shell_join_ints(seeds)}"',
            "PRECHECK_SMOKE=0",
        ]
    )

    stage1 = _env_command_block(
        shared_env_lines
        + [
            'PRUNING_VARIANTS="path-magnitude magnitude magnitude-rescaled"',
            'PRUNING_RATIOS="0.1 0.2 0.4 0.6 0.8"',
        ],
        "bash repro/icml25/utils/run_campaign.sh",
    )
    stage2 = _env_command_block(
        shared_env_lines
        + [
            'PRUNING_VARIANTS="path-magnitude-rescaled"',
            'PRUNING_RATIOS="0.4"',
        ],
        "bash repro/icml25/utils/run_campaign.sh",
    )

    return textwrap.dedent(
        "\n".join(
            [
                "# Stage 1: table variants",
                stage1,
                "",
                "# Stage 2: 40% path-magnitude-rescaled curves",
                stage2,
            ]
        )
    ).strip()
