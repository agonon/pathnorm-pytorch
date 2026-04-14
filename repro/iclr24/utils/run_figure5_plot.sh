#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="repro/iclr24/results/rerun"
num_epochs="${NUM_EPOCHS:-90}"
num_seeds="${NUM_SEEDS:-3}"
size_datasets="${SIZE_DATASETS:-39636 79272 158544 317089 634178}"

lr="${LR:-0.1}"
wd="${WD:-0.0001}"
lr_scheduler="${LR_SCHEDULER:-multi-step}"
imp_iters="${IMP_ITERS:-0}"
arch="${ARCH:-resnet18}"
size_dataset="${SIZE_DATASET:-0}"
seed="${SEED:-0}"
percentage_pruning="${PERCENTAGE_PRUNING:-}"
results_training_dir="${RESULTS_TRAINING_DIR:-${results_root}/4_train_increasing_dataset/seed=${seed}/${arch}/size_dataset=${size_dataset}/lr=${lr}_wd=${wd}_epochs=${num_epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}}"
saving_dir="${SAVING_DIR:-${results_root}/5_plot_increasing_dataset/num_seeds=${num_seeds}/${arch}/lr=${lr}_wd=${wd}_epochs=${num_epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}}"
rank="${RANK:-0}"

exec python -m repro.iclr24.utils.plot_increasing_dataset \
    --num-epochs "$num_epochs" \
    --num-seeds "$num_seeds" \
    --saving-dir "$saving_dir" \
    --results-training-dir "$results_training_dir" \
    --size-datasets ${size_datasets} \
    --rank "$rank"
