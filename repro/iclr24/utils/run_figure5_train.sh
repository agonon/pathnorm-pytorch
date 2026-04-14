#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="repro/iclr24/results/rerun"
workers="${WORKERS:-16}"
batch_size="${BATCH_SIZE:-1024}"
epochs="${EPOCHS:-1}"
lr="${LR:-0.1}"
wd="${WD:-0.0001}"
lr_scheduler="${LR_SCHEDULER:-multi-step}"
imp_iters="${IMP_ITERS:-0}"
percentage_pruning="${PERCENTAGE_PRUNING:-}"
arch="${ARCH:-resnet18}"
data_dir="${DATA_DIR:-/path/to/ImageNet/}"
saving_root="${SAVING_ROOT:-${results_root}/4_train_increasing_dataset}"
seeds="${SEEDS:-0 1 2}"
size_datasets="${SIZE_DATASETS:-39636 79272 158544 317089 634178}"
use_tensorboard="${USE_TENSORBOARD:-0}"
distributed="${DISTRIBUTED:-0}"
world_size="${WORLD_SIZE:-1}"
rank="${RANK:-0}"
dist_url="${DIST_URL:-tcp://127.0.0.1:23456}"
gpu="${GPU:-}"

read -r -a seed_array <<< "${seeds}"
read -r -a size_array <<< "${size_datasets}"
total_runs=$((${#seed_array[@]} * ${#size_array[@]}))
completed_runs=0
wall_start=$(date +%s)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 5 training launch"
echo "  saving_root=${saving_root}"
echo "  data_dir=${data_dir}"
echo "  total_runs=${total_runs} (seeds=${#seed_array[@]} x dataset_sizes=${#size_array[@]})"

for seed in ${seeds}
do
    for size_dataset in ${size_datasets}
    do
        current_run=$((completed_runs + 1))
        saving_dir="${saving_root}/seed=${seed}/${arch}/size_dataset=${size_dataset}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 5 run ${current_run}/${total_runs}: seed=${seed}, size_dataset=${size_dataset}"

        extra_args=()
        if [[ "${use_tensorboard}" == "1" ]]; then
            extra_args+=(--tensorboard)
        fi
        if [[ "${distributed}" == "1" ]]; then
            extra_args+=(--multiprocessing-distributed --world-size "${world_size}" --rank "${rank}" --dist-url "${dist_url}")
            if [[ -n "${gpu}" ]]; then
                extra_args+=(--gpu "${gpu}")
            fi
        else
            extra_args+=(--no-data-parallel)
        fi

        python -m repro.iclr24.utils.train_imagenet "$data_dir" \
            --arch "$arch" \
            --epochs "$epochs" \
            --workers "$workers" \
            --batch-size "$batch_size" \
            --lr "$lr" \
            --wd "$wd" \
            --lr-scheduler "$lr_scheduler" \
            --saving-dir "$saving_dir" \
            --IMP-iters "$imp_iters" \
            --test-after-train \
            --size-dataset "$size_dataset" \
            --seed "$seed" \
            "${extra_args[@]}"

        completed_runs=$((completed_runs + 1))
        elapsed=$(( $(date +%s) - wall_start ))
        average_run=$(( elapsed / completed_runs ))
        remaining_runs=$(( total_runs - completed_runs ))
        eta_seconds=$(( average_run * remaining_runs ))
        printf '[%(%Y-%m-%d %H:%M:%S)T] Figure 5 progress: completed %d/%d runs, elapsed %02dh:%02dm:%02ds, ETA %02dh:%02dm:%02ds\n' \
            -1 \
            "${completed_runs}" \
            "${total_runs}" \
            $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60)) \
            $((eta_seconds / 3600)) $(((eta_seconds % 3600) / 60)) $((eta_seconds % 60))
    done
done
