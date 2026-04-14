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
variants="${VARIANTS:-path-magnitude path-magnitude-rescaled magnitude magnitude-rescaled}"
pruning_ratios="${PRUNING_RATIOS:-0.1 0.2 0.4 0.6 0.8}"

format_duration() {
    local total_seconds="$1"
    local hours=$(( total_seconds / 3600 ))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$(( total_seconds % 60 ))
    printf '%02dh:%02dm:%02ds' "${hours}" "${minutes}" "${seconds}"
}

read -r -a seed_array <<< "${seeds}"
read -r -a variant_array <<< "${variants}"
read -r -a pruning_ratio_array <<< "${pruning_ratios}"
total_runs=$(( ${#seed_array[@]} * ${#variant_array[@]} * ${#pruning_ratio_array[@]} ))
completed_runs=0
SECONDS=0

for seed in "${seed_array[@]}"
do
    for variant in "${variant_array[@]}"
    do
        for pruning_ratio in "${pruning_ratio_array[@]}"
        do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Accuracy sweep run $((completed_runs + 1))/${total_runs}: seed=${seed}, variant=${variant}, pruning_ratio=${pruning_ratio}"
            SEED="${seed}" \
            ARCH="${arch}" \
            VARIANT="${variant}" \
            PRUNING_RATIO="${pruning_ratio}" \
            EPOCHS="${epochs}" \
            LR="${lr}" \
            WD="${wd}" \
            LR_SCHEDULER="${lr_scheduler}" \
            WORKERS="${workers}" \
            BATCH_SIZE="${batch_size}" \
            DATA_DIR="${data_dir}" \
            RESULTS_ROOT="${results_root}" \
            OVERWRITE="${OVERWRITE:-}" \
            bash repro/icml25/utils/run_pruning_experiment.sh

            completed_runs=$((completed_runs + 1))
            elapsed_seconds="${SECONDS}"
            if [[ "${completed_runs}" -lt "${total_runs}" ]]; then
                eta_seconds=$(( elapsed_seconds * (total_runs - completed_runs) / completed_runs ))
                echo "Accuracy sweep progress: completed ${completed_runs}/${total_runs}, elapsed $(format_duration "${elapsed_seconds}"), ETA $(format_duration "${eta_seconds}")"
            else
                echo "Accuracy sweep progress: completed ${completed_runs}/${total_runs}, elapsed $(format_duration "${elapsed_seconds}")"
            fi
        done
    done
done

python -m repro.icml25.utils.aggregate_results --results-root "${results_root}"
