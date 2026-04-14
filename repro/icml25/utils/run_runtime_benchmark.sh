#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="${RESULTS_ROOT:-repro/icml25/results/rerun}"
saving_path="${SAVING_PATH:-${results_root}/benchmark_summary.csv}"
repeats="${REPEATS:-10}"

exec python -m repro.icml25.utils.benchmark \
    --saving-path "$saving_path" \
    --repeats "$repeats"
