#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

results_root="${RESULTS_ROOT:-repro/icml25/results}"
rerun_root="${results_root}/rerun"
paper_release_root="${results_root}/paper_release"

required_artifacts=(
    benchmark_summary.csv
    dense_summary.csv
    pruning_accuracy_body.csv
    training_curves_0.4.json
    pruning_accuracy_full.csv
)
optional_artifacts=(
    plot_training_curves_test0.4.pdf
    plot_training_curves_test0.4.png
)

mkdir -p "${paper_release_root}"

missing=()
for artifact in "${required_artifacts[@]}"; do
    if [[ ! -f "${rerun_root}/${artifact}" ]]; then
        missing+=("${artifact}")
    fi
done

if (( ${#missing[@]} > 0 )); then
    printf 'Cannot update results/paper_release. Missing rerun outputs:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

for artifact in "${required_artifacts[@]}"; do
    cp "${rerun_root}/${artifact}" "${paper_release_root}/${artifact}"
    echo "Published ${paper_release_root}/${artifact}"
done

for artifact in "${optional_artifacts[@]}"; do
    if [[ -f "${rerun_root}/${artifact}" ]]; then
        cp "${rerun_root}/${artifact}" "${paper_release_root}/${artifact}"
        echo "Published ${paper_release_root}/${artifact}"
    fi
done

echo "Updated results/paper_release from rerun outputs."
