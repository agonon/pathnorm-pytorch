#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

mode="${MODE:-canonical}"
data_dir="${DATA_DIR:-/path/to/ImageNet/}"
results_root="${RESULTS_ROOT:-repro/icml25/results/rerun}"
overwrite="${OVERWRITE:-0}"
gpu_ids_raw="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
runtime_repeats="${RUNTIME_REPEATS:-10}"
parallel_jobs="${PARALLEL_JOBS:-}"
precheck_smoke="${PRECHECK_SMOKE:-0}"
require_release_gate="${REQUIRE_RELEASE_GATE:-1}"

dense_seeds="${DENSE_SEEDS:-1 2 3}"
dense_epochs="${DENSE_EPOCHS:-90}"
dense_workers="${DENSE_WORKERS:-16}"
dense_batch_size="${DENSE_BATCH_SIZE:-1024}"

pruning_seeds="${PRUNING_SEEDS:-${dense_seeds}}"
pruning_variants="${PRUNING_VARIANTS:-path-magnitude path-magnitude-rescaled magnitude magnitude-rescaled}"
pruning_ratios="${PRUNING_RATIOS:-0.1 0.2 0.4 0.6 0.8}"
pruning_epochs="${PRUNING_EPOCHS:-90}"
pruning_workers="${PRUNING_WORKERS:-8}"
pruning_batch_size="${PRUNING_BATCH_SIZE:-1024}"

if [[ "${mode}" == "smoke" ]]; then
    runtime_repeats="${RUNTIME_REPEATS:-2}"
    dense_seeds="${DENSE_SEEDS:-1}"
    dense_epochs="${DENSE_EPOCHS:-6}"
    dense_workers="${DENSE_WORKERS:-8}"
    pruning_seeds="${PRUNING_SEEDS:-1}"
    pruning_variants="${PRUNING_VARIANTS:-path-magnitude}"
    pruning_ratios="${PRUNING_RATIOS:-0.4}"
    pruning_epochs="${PRUNING_EPOCHS:-6}"
    pruning_workers="${PRUNING_WORKERS:-8}"
    parallel_jobs="${PARALLEL_JOBS:-1}"
    require_release_gate="${REQUIRE_RELEASE_GATE:-0}"
fi

normalize_gpu_ids() {
    local normalized="${gpu_ids_raw//,/ }"
    normalized="${normalized#"${normalized%%[![:space:]]*}"}"
    normalized="${normalized%"${normalized##*[![:space:]]}"}"
    if [[ -n "${normalized}" ]]; then
        echo "${normalized}"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd' ' -
        return
    fi

    echo "0"
}

format_duration() {
    local total_seconds="$1"
    local days=$(( total_seconds / 86400 ))
    local hours=$(( (total_seconds % 86400) / 3600 ))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$(( total_seconds % 60 ))
    if (( days > 0 )); then
        printf '%dd %02dh %02dm' "${days}" "${hours}" "${minutes}"
    elif (( hours > 0 )); then
        printf '%02dh %02dm %02ds' "${hours}" "${minutes}" "${seconds}"
    elif (( minutes > 0 )); then
        printf '%02dm %02ds' "${minutes}" "${seconds}"
    else
        printf '%02ds' "${seconds}"
    fi
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

read -r -a gpu_array <<< "$(normalize_gpu_ids)"
if [[ "${#gpu_array[@]}" -eq 0 ]]; then
    echo "Could not resolve any GPU ids."
    exit 1
fi

if [[ -z "${parallel_jobs}" ]]; then
    parallel_jobs="${#gpu_array[@]}"
fi
if (( parallel_jobs < 1 )); then
    parallel_jobs=1
fi
if (( parallel_jobs > ${#gpu_array[@]} )); then
    parallel_jobs="${#gpu_array[@]}"
fi

log_root="${results_root}/logs/campaign"
mkdir -p "${log_root}"

declare -a free_gpus=("${gpu_array[@]:0:${parallel_jobs}}")
declare -a active_pids=()
declare -A pid_to_gpu=()
declare -A pid_to_label=()
declare -A pid_to_log=()

remove_pid() {
    local finished_pid="$1"
    local updated=()
    local pid
    for pid in "${active_pids[@]:-}"; do
        if [[ "${pid}" != "${finished_pid}" ]]; then
            updated+=("${pid}")
        fi
    done
    active_pids=("${updated[@]}")
}

kill_active_jobs() {
    local pid
    for pid in "${active_pids[@]:-}"; do
        kill "${pid}" >/dev/null 2>&1 || true
    done
}

reap_one() {
    while true; do
        local finished_pid=""
        local pid
        for pid in "${active_pids[@]:-}"; do
            if ! kill -0 "${pid}" >/dev/null 2>&1; then
                finished_pid="${pid}"
                break
            fi
        done

        if [[ -z "${finished_pid}" ]]; then
            sleep 5
            continue
        fi

        local exit_code=0
        if wait "${finished_pid}"; then
            exit_code=0
        else
            exit_code=$?
        fi

        local gpu="${pid_to_gpu[${finished_pid}]}"
        local label="${pid_to_label[${finished_pid}]}"
        local log_path="${pid_to_log[${finished_pid}]}"
        free_gpus+=("${gpu}")
        remove_pid "${finished_pid}"
        unset "pid_to_gpu[${finished_pid}]"
        unset "pid_to_label[${finished_pid}]"
        unset "pid_to_log[${finished_pid}]"

        if (( exit_code != 0 )); then
            echo "[$(timestamp)] Task failed: ${label}"
            echo "Last lines from ${log_path}:"
            tail -n 60 "${log_path}" || true
            kill_active_jobs
            exit "${exit_code}"
        fi

        echo "[$(timestamp)] Task finished: ${label}"
        return
    done
}

wait_for_slot() {
    while (( ${#free_gpus[@]} == 0 )); do
        reap_one
    done
}

launch_task() {
    local label="$1"
    local command="$2"
    local log_path="$3"
    mkdir -p "$(dirname "${log_path}")"
    wait_for_slot

    local gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")

    echo "[$(timestamp)] Launching ${label} on GPU ${gpu}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        exec bash -lc "${command}"
    ) >"${log_path}" 2>&1 &

    local pid=$!
    active_pids+=("${pid}")
    pid_to_gpu["${pid}"]="${gpu}"
    pid_to_label["${pid}"]="${label}"
    pid_to_log["${pid}"]="${log_path}"
}

wait_for_all_tasks() {
    while (( ${#active_pids[@]} > 0 )); do
        reap_one
    done
}

run_runtime_stage() {
    local stage_log="${log_root}/runtime.log"
    local benchmark_path="${results_root}/benchmark_summary.csv"
    if [[ -f "${benchmark_path}" && "${overwrite}" != "1" ]]; then
        echo "[$(timestamp)] Runtime benchmark already exists at ${benchmark_path}"
        return
    fi

    echo "[$(timestamp)] Starting runtime benchmark stage"
    (
        export CUDA_VISIBLE_DEVICES="${gpu_array[0]}"
        RESULTS_ROOT="${results_root}" REPEATS="${runtime_repeats}" \
            bash repro/icml25/utils/run_runtime_benchmark.sh
    ) >"${stage_log}" 2>&1
    echo "[$(timestamp)] Runtime benchmark finished"
}

run_dense_stage() {
    echo "[$(timestamp)] Starting dense stage for seeds: ${dense_seeds}"
    local seed
    for seed in ${dense_seeds}; do
        local dense_dir="${results_root}/dense/seed=${seed}/resnet18/lr=0.1_wd=0.0001_epochs=${dense_epochs}_scheduler=multi-step"
        local log_path="${log_root}/dense-seed=${seed}.log"
        local command
        command="DATA_DIR='${data_dir}' RESULTS_ROOT='${results_root}' SEEDS='${seed}' EPOCHS='${dense_epochs}' WORKERS='${dense_workers}' BATCH_SIZE='${dense_batch_size}' OVERWRITE='${overwrite}' bash repro/icml25/utils/run_dense_train.sh"
        if [[ -f "${dense_dir}/model_best.pth.tar" && "${overwrite}" != "1" ]]; then
            echo "[$(timestamp)] Dense seed ${seed} already complete at ${dense_dir}"
            continue
        fi
        launch_task "dense-seed=${seed}" "${command}" "${log_path}"
    done
    wait_for_all_tasks
    python -m repro.icml25.utils.aggregate_results --results-root "${results_root}"
    echo "[$(timestamp)] Dense stage finished"
}

run_pruning_stage() {
    echo "[$(timestamp)] Starting pruning stage"
    local seed
    local variant
    local pruning_ratio
    for seed in ${pruning_seeds}; do
        for variant in ${pruning_variants}; do
            for pruning_ratio in ${pruning_ratios}; do
                local run_dir="${results_root}/pruning/seed=${seed}/resnet18/variant=${variant}/pruning_ratio=${pruning_ratio}"
                local log_path="${log_root}/pruning-seed=${seed}-variant=${variant}-ratio=${pruning_ratio}.log"
                local command
                command="DATA_DIR='${data_dir}' RESULTS_ROOT='${results_root}' SEED='${seed}' VARIANT='${variant}' PRUNING_RATIO='${pruning_ratio}' EPOCHS='${pruning_epochs}' WORKERS='${pruning_workers}' BATCH_SIZE='${pruning_batch_size}' OVERWRITE='${overwrite}' bash repro/icml25/utils/run_pruning_experiment.sh"
                if [[ -f "${run_dir}/rank=0/csv/results.csv" && "${overwrite}" != "1" ]]; then
                    echo "[$(timestamp)] Pruning run already complete at ${run_dir}"
                    continue
                fi
                launch_task "pruning-seed=${seed}-variant=${variant}-ratio=${pruning_ratio}" "${command}" "${log_path}"
            done
        done
    done
    wait_for_all_tasks
    python -m repro.icml25.utils.aggregate_results --results-root "${results_root}"
    echo "[$(timestamp)] Pruning stage finished"
}

run_release_gate_check() {
    if [[ "${require_release_gate}" != "1" ]]; then
        return
    fi
    python - <<'PY' "${results_root}"
from pathlib import Path
import sys

results_root = Path(sys.argv[1])
required = [
    "benchmark_summary.csv",
    "dense_summary.csv",
    "pruning_accuracy_body.csv",
    "training_curves_0.4.json",
    "pruning_accuracy_full.csv",
]
missing = [name for name in required if not (results_root / name).exists()]
if missing:
    raise SystemExit(
        "Release gate failed. Missing rerun artifacts: " + ", ".join(missing)
    )
print("Release gate passed for rerun artifacts.")
PY
}

run_smoke_precheck() {
    local smoke_root="${results_root}/smoke_precheck"
    echo "[$(timestamp)] Starting smoke precheck under ${smoke_root}"
    MODE=smoke \
    DATA_DIR="${data_dir}" \
    RESULTS_ROOT="${smoke_root}" \
    GPU_IDS="${gpu_array[0]}" \
    OVERWRITE="${overwrite}" \
    PRECHECK_SMOKE=0 \
    REQUIRE_RELEASE_GATE=0 \
    bash repro/icml25/utils/run_campaign.sh
    echo "[$(timestamp)] Smoke precheck finished"
}

echo "[$(timestamp)] ICML campaign mode=${mode}"
echo "[$(timestamp)] results_root=${results_root}"
echo "[$(timestamp)] gpu_ids=${gpu_array[*]}"
echo "[$(timestamp)] parallel_jobs=${parallel_jobs}"

if [[ "${precheck_smoke}" == "1" ]]; then
    run_smoke_precheck
fi

run_runtime_stage
run_dense_stage
run_pruning_stage
run_release_gate_check

echo "[$(timestamp)] ICML campaign completed successfully"
