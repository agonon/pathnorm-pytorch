#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="${RESULTS_ROOT:-repro/icml25/results/rerun}"
seed="${SEED:-1}"
arch="${ARCH:-resnet18}"
variant="${VARIANT:-path-magnitude}"
pruning_ratio="${PRUNING_RATIO:-0.4}"
epochs="${EPOCHS:-90}"
lr="${LR:-0.1}"
wd="${WD:-0.0001}"
lr_scheduler="${LR_SCHEDULER:-multi-step}"
workers="${WORKERS:-16}"
batch_size="${BATCH_SIZE:-1024}"
data_dir="${DATA_DIR:-/path/to/ImageNet/}"
dense_dir="${DENSE_DIR:-${results_root}/dense/seed=${seed}/${arch}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}}"
run_dir="${RUN_DIR:-${results_root}/pruning/seed=${seed}/${arch}/variant=${variant}/pruning_ratio=${pruning_ratio}}"

python -m repro.icml25.utils.prepare_pruned_checkpoint \
    --dense-dir "$dense_dir" \
    --run-dir "$run_dir" \
    --variant "$variant" \
    --pruning-ratio "$pruning_ratio" \
    --seed "$seed" \
    ${OVERWRITE:+--overwrite}

exec python -m repro.icml25.utils.finetune_pruned "$data_dir" \
    --dense-dir "$dense_dir" \
    --run-dir "$run_dir" \
    --epochs "$epochs" \
    --workers "$workers" \
    --batch-size "$batch_size" \
    --lr "$lr" \
    --wd "$wd" \
    --lr-scheduler "$lr_scheduler" \
    --seed "$seed" \
    ${OVERWRITE:+--overwrite}
