#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

data_dir="${DATA_DIR:-/path/to/ImageNet/}"
batch_size="${BATCH_SIZE:-1024}"
workers="${WORKERS:-16}"
B_value="${B_VALUE:-2.640000104904175}"

base_args=(
    --batch-size "$batch_size"
    --workers "$workers"
)

if [[ -n "${data_dir}" ]]; then
    base_args+=(--data-dir "$data_dir")
fi

extra_args=()
if [[ -n "${B_value}" ]]; then
    extra_args+=(--B "$B_value")
fi

exec python -m repro.iclr24.utils.compute_table2 \
    "${base_args[@]}" \
    "${extra_args[@]}"
