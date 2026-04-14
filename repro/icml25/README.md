# ICML 2025 Reproduction

This folder contains the reproduction material for [*A Rescaling-Invariant Lipschitz Bound Based on Path-Metrics for Modern ReLU Network Parameterizations*](https://arxiv.org/pdf/2405.15006), by Antoine Gonon, Nicolas Brisebarre, Elisa Riccietti, and Rémi Gribonval (ICML 2025).

Notebook: [icml25_repro.ipynb](icml25_repro.ipynb)

## Contents

- [icml25_repro.ipynb](icml25_repro.ipynb), the main notebook;
- [`utils/`](utils), the public helper scripts and Python modules;
- [`results/`](results), the included results and the default location for fresh reruns.

The notebook follows the paper:

- runtime comparison of pruning criteria;
- proof-of-concept pruning setup on ResNet-18 / ImageNet;
- dense baseline;
- main pruning table;
- 40% fine-tuning curves;
- appendix pruning table.

By default, the notebook reads the results included in `results/paper_release/`. The included pruning results use one seed. To inspect your own runs, point it to `results/rerun/`.

## Installation

From the repository root:

```bash
pip install -e .[repro]
```

## Commands

Runtime benchmark:

```bash
RESULTS_ROOT=repro/icml25/results/rerun \
bash repro/icml25/utils/run_runtime_benchmark.sh
```

Dense baseline:

```bash
DATA_DIR=data/imagenet \
RESULTS_ROOT=repro/icml25/results/rerun \
SEEDS="1 2 3" \
bash repro/icml25/utils/run_dense_train.sh
```

Pruning rerun (1 seed):

```bash
DATA_DIR=data/imagenet \
RESULTS_ROOT=repro/icml25/results/rerun \
DENSE_SEEDS="1" \
PRUNING_SEEDS="1" \
bash repro/icml25/utils/run_campaign.sh
```

Pruning rerun (3 seeds):

```bash
DATA_DIR=data/imagenet \
RESULTS_ROOT=repro/icml25/results/rerun \
DENSE_SEEDS="1 2 3" \
PRUNING_SEEDS="1 2 3" \
bash repro/icml25/utils/run_campaign.sh
```

These runs produce the notebook files:

- `benchmark_summary.csv`
- `dense_summary.csv`
- `pruning_accuracy_body.csv`
- `training_curves_0.4.json`
- `plot_training_curves_test0.4.pdf`
- `plot_training_curves_test0.4.png`
- `pruning_accuracy_full.csv`

Once the rerun artifacts you want to ship are complete, you can promote them into `results/paper_release/` with:

```bash
bash repro/icml25/utils/publish_paper_release.sh
```

## Citation

If you use this reproduction material, please cite:

```bibtex
@inproceedings{gonon2025pathmetrics,
  title={A Rescaling-Invariant Lipschitz Bound Based on Path-Metrics for Modern ReLU Network Parameterizations},
  author={Gonon, Antoine and Brisebarre, Nicolas and Riccietti, Elisa and Gribonval, R{\'e}mi},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

If you use the core path-norm implementation as well, please also cite the ICLR 2024 toolkit paper.
