#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="repro/iclr24/results/rerun"
num_seeds="${NUM_SEEDS:-1}"

arch="${ARCH:-resnet18}"
epochs="${EPOCHS:-90}"
lr="${LR:-0.1}"
wd="${WD:-0.0001}"
lr_scheduler="${LR_SCHEDULER:-multi-step}"
percentage_pruning="${PERCENTAGE_PRUNING:-0.2}"
imp_iters="${IMP_ITERS:-20}"
seed="${SEED:-0}"
results_training_dir="${RESULTS_TRAINING_DIR:-${results_root}/2_train_imp/seed=${seed}/${arch}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}/}"
saving_dir="${SAVING_DIR:-${results_root}/3_plot_imp/num_seeds=${num_seeds}/${arch}/lr=${lr}_wd=${wd}_epochs=${epochs}_scheduler=${lr_scheduler}_percentage_pruning=${percentage_pruning}_imp_iters=${imp_iters}/}"
rank="${RANK:--1}"

exec python -m repro.iclr24.utils.plot_imp \
    --num-seeds "$num_seeds" \
    --saving-dir "$saving_dir" \
    --results-training-dir "$results_training_dir" \
    --rank "$rank"
