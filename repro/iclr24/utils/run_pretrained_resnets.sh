#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="repro/iclr24/results/rerun"
saving_dir="${SAVING_DIR:-${results_root}/1_compute_pretrained_pathnorm_margins_resnets/}"
data_dir="${DATA_DIR:-/path/to/ImageNet/}"
batch_size="${BATCH_SIZE:-1024}"
workers="${WORKERS:-16}"
margins_already_computed="${MARGINS_ALREADY_COMPUTED:-1}"

base_args=(
    --saving-dir "$saving_dir"
    --batch-size "$batch_size"
    --workers "$workers"
)

if [[ -n "${data_dir}" ]]; then
    base_args+=(--data-dir "$data_dir")
fi

extra_args=()
if [[ "${margins_already_computed}" == "1" ]]; then
    extra_args+=(--margins-already-computed)
fi

exec python -m repro.iclr24.utils.pretrained_resnets \
    "${base_args[@]}" \
    "${extra_args[@]}"
