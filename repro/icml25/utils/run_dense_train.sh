#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="${RESULTS_ROOT:-repro/icml25/results/rerun}"
seeds="${SEEDS:-${SEED:-1 2 3}}"
arch="${ARCH:-resnet18}"
epochs="${EPOCHS:-90}"
lr="${LR:-0.1}"
wd="${WD:-0.0001}"
lr_scheduler="${LR_SCHEDULER:-multi-step}"
workers="${WORKERS:-16}"
batch_size="${BATCH_SIZE:-1024}"
data_dir="${DATA_DIR:-/path/to/ImageNet/}"
explicit_dense_dir="${DENSE_DIR:-}"

format_duration() {
    local total_seconds="$1"
    local hours=$(( total_seconds / 3600 ))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$(( total_seconds % 60 ))
    printf '%02dh:%02dm:%02ds' "${hours}" "${minutes}" "${seconds}"
}

read -r -a seed_array <<< "${seeds}"
total_runs="${#seed_array[@]}"
completed_runs=0
SECONDS=0

for seed in "${seed_array[@]}"
do
    if [[ -n "${explicit_dense_dir}" && "${seeds}" != "${seed}" ]]; then
        echo "DENSE_DIR can only be used with a single seed."
        exit 1
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dense run $((completed_runs + 1))/${total_runs}: seed=${seed}"
    dense_dir="${explicit_dense_dir:-${results_root}/dense/seed=${seed}/${arch}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}}"

    if [[ -f "${dense_dir}/model_best.pth.tar" && "${OVERWRITE:-0}" != "1" ]]; then
        echo "Dense run already exists at ${dense_dir}"
        completed_runs=$((completed_runs + 1))
        elapsed_seconds="${SECONDS}"
        if [[ "${completed_runs}" -gt 0 && "${completed_runs}" -lt "${total_runs}" ]]; then
            eta_seconds=$(( elapsed_seconds * (total_runs - completed_runs) / completed_runs ))
            echo "Dense progress: completed ${completed_runs}/${total_runs}, elapsed $(format_duration "${elapsed_seconds}"), ETA $(format_duration "${eta_seconds}")"
        else
            echo "Dense progress: completed ${completed_runs}/${total_runs}, elapsed $(format_duration "${elapsed_seconds}")"
        fi
        continue
    fi

    python -m repro.icml25.utils.train_imagenet "$data_dir" \
        --arch "$arch" \
        --epochs "$epochs" \
        --workers "$workers" \
        --batch-size "$batch_size" \
        --lr "$lr" \
        --wd "$wd" \
        --lr-scheduler "$lr_scheduler" \
        --saving-dir "$dense_dir" \
        --test-after-train \
        --seed "$seed"

    completed_runs=$((completed_runs + 1))
    elapsed_seconds="${SECONDS}"
    if [[ "${completed_runs}" -lt "${total_runs}" ]]; then
        eta_seconds=$(( elapsed_seconds * (total_runs - completed_runs) / completed_runs ))
        echo "Dense progress: completed ${completed_runs}/${total_runs}, elapsed $(format_duration "${elapsed_seconds}"), ETA $(format_duration "${eta_seconds}")"
    else
        echo "Dense progress: completed ${completed_runs}/${total_runs}, elapsed $(format_duration "${elapsed_seconds}")"
    fi
done

python -m repro.icml25.utils.aggregate_results --results-root "${results_root}"
