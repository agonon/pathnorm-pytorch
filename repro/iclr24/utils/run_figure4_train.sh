#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="repro/iclr24/results/rerun"
epochs="${EPOCHS:-7}"
lr="${LR:-0.1}"
wd="${WD:-0.0001}"
lr_scheduler="${LR_SCHEDULER:-multi-step}"
imp_iters="${IMP_ITERS:-2}"
percentage_pruning="${PERCENTAGE_PRUNING:-0.2}"
start_imp_iter="${START_IMP_ITER:-0}"
seed="${SEED:-0}"
arch="${ARCH:-resnet18}"
workers="${WORKERS:-16}"
batch_size="${BATCH_SIZE:-1024}"
data_dir="${DATA_DIR:-/path/to/ImageNet/}"
saving_dir="${SAVING_DIR:-${results_root}/2_train_imp/seed=${seed}/${arch}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}}"
distributed="${DISTRIBUTED:-0}"
world_size="${WORLD_SIZE:-1}"
rank="${RANK:-0}"
dist_url="${DIST_URL:-tcp://127.0.0.1:23456}"
gpu="${GPU:-}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Figure 4 training launch"
echo "  saving_dir=${saving_dir}"
echo "  data_dir=${data_dir}"
echo "  distributed=${distributed}"
echo "  epochs=${epochs} imp_iters=${imp_iters} percentage_pruning=${percentage_pruning}"

extra_args=()
if [[ "${EVALUATE_BEFORE_TRAIN:-0}" == "1" ]]; then
    extra_args+=(--evaluate-before-train)
fi
if [[ -n "${RESUME_PATH:-}" ]]; then
    extra_args+=(--resume "${RESUME_PATH}")
fi
if [[ "${USE_TENSORBOARD:-0}" == "1" ]]; then
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

exec python -m repro.iclr24.utils.train_imagenet "$data_dir" \
    --arch "$arch" \
    --epochs "$epochs" \
    --workers "$workers" \
    --batch-size "$batch_size" \
    --lr "$lr" \
    --wd "$wd" \
    --lr-scheduler "$lr_scheduler" \
    --saving-dir "$saving_dir" \
    --IMP-iters "$imp_iters" \
    --percentage-pruning "$percentage_pruning" \
    --test-after-train \
    --seed "$seed" \
    --start-IMP-iter "$start_imp_iter" \
    "${extra_args[@]}"
