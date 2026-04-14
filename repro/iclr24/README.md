# ICLR 2024 Reproduction

This folder contains the reproduction material for [*A path-norm toolkit for modern networks: consequences, promises and challenges*](https://arxiv.org/abs/2310.01225), by Antoine Gonon, Nicolas Brisebarre, Elisa Riccietti, and Rémi Gribonval (ICLR Spotlight 2024).

Notebook: [iclr24_repro.ipynb](iclr24_repro.ipynb)

## What Is Here

- [iclr24_repro.ipynb](iclr24_repro.ipynb), the main notebook;
- [`utils/`](utils), the public helper scripts and Python modules;
- [`results/`](results), the included results and the default location for fresh reruns.

The notebook reproduces:

- Table 2: the first part of the ImageNet bound;
- pretrained ResNet path-norms and margin statistics;
- Figure 4: iterative magnitude pruning on ResNet-18;
- Figure 5: training on increasing subsets of ImageNet.

By default, the notebook reads the results included in `results/paper_release/`. In the included bundle, Figure 4 uses `40` epochs and `8` pruning iterations, and Figure 5 uses `1` seed. To inspect your own runs, point it to `results/rerun/...`.

## Installation

From the repository root:

```bash
pip install -e .[repro]
```

## ImageNet Data

The pretrained ResNet section and the Figure 4 / Figure 5 reruns require a prepared ImageNet directory.

Option 1: download from the gated Hugging Face mirror after accepting access on [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k):

```bash
HF_TOKEN=... python -m repro.iclr24.utils.download_imagenet_from_hf \
  --streaming \
  --dest-dir data/imagenet \
  --cache-dir .hf-cache/imagenet
```

Option 2: stage an existing prepared copy:

```bash
python -m repro.iclr24.utils.prepare_imagenet_data \
  --source-dir /path/to/already-prepared/imagenet \
  --dest-dir data/imagenet
```

Verification:

```bash
python -m repro.iclr24.utils.verify_imagenet_data data/imagenet
```

## Commands

Practical reruns:

```bash
DATA_DIR=data/imagenet \
B_VALUE=2.640000104904175 \
bash repro/iclr24/utils/run_table2.sh
```

```bash
DATA_DIR=data/imagenet \
bash repro/iclr24/utils/run_pretrained_resnets.sh
```

Figure 4 rerun (40 epochs, 8 pruning iterations):

```bash
DATA_DIR=data/imagenet \
SAVING_DIR=repro/iclr24/results/rerun/mid_scale/2_train_imp/seed=0/resnet18/lr=0.1_wd=0.0001_epochs=40_scheduler=multi-step_percentage_pruning=0.2_imp_iters=8 \
EPOCHS=40 \
IMP_ITERS=8 \
PERCENTAGE_PRUNING=0.2 \
bash repro/iclr24/utils/run_figure4_train.sh
```

```bash
RESULTS_TRAINING_DIR=repro/iclr24/results/rerun/mid_scale/2_train_imp/seed=0/resnet18/lr=0.1_wd=0.0001_epochs=40_scheduler=multi-step_percentage_pruning=0.2_imp_iters=8 \
SAVING_DIR=repro/iclr24/results/rerun/mid_scale/3_plot_imp/num_seeds=1/resnet18/lr=0.1_wd=0.0001_epochs=40_scheduler=multi-step_percentage_pruning=0.2_imp_iters=8 \
EPOCHS=40 \
IMP_ITERS=8 \
PERCENTAGE_PRUNING=0.2 \
bash repro/iclr24/utils/run_figure4_plot.sh
```

Figure 5 rerun (1 seed):

```bash
DATA_DIR=data/imagenet \
EPOCHS=90 \
SAVING_ROOT=repro/iclr24/results/rerun/mid_scale/4_train_increasing_dataset \
SEEDS="0" \
SIZE_DATASETS="39636 79272 158544 317089 634178" \
bash repro/iclr24/utils/run_figure5_train.sh
```

```bash
RESULTS_TRAINING_DIR=repro/iclr24/results/rerun/mid_scale/4_train_increasing_dataset/seed=0/resnet18/size_dataset=39636/lr=0.1_wd=0.0001_epochs=90_scheduler=multi-step_percentage_pruning=_imp_iters=0 \
SAVING_DIR=repro/iclr24/results/rerun/mid_scale/5_plot_increasing_dataset/num_seeds=1/resnet18/lr=0.1_wd=0.0001_epochs=90_scheduler=multi-step_percentage_pruning=_imp_iters=0 \
NUM_EPOCHS=90 \
NUM_SEEDS=1 \
bash repro/iclr24/utils/run_figure5_plot.sh
```

Figure 4 rerun (90 epochs, 20 pruning iterations):

```bash
DATA_DIR=data/imagenet \
EPOCHS=90 \
IMP_ITERS=20 \
PERCENTAGE_PRUNING=0.2 \
bash repro/iclr24/utils/run_figure4_train.sh
```

Figure 5 rerun (3 seeds):

```bash
DATA_DIR=data/imagenet \
EPOCHS=90 \
SEEDS="0 1 2" \
SIZE_DATASETS="39636 79272 158544 317089 634178" \
bash repro/iclr24/utils/run_figure5_train.sh
```

Table 2 runs in minutes. The pretrained ResNet rerun takes a few hours on one GPU. Figure 4 and Figure 5 are long offline jobs.

## Citation

If you use this reproduction material, please cite:

```bibtex
@inproceedings{gonon2024pathnorm,
  title={A path-norm toolkit for modern networks: consequences, promises and challenges},
  author={Gonon, Antoine and Brisebarre, Nicolas and Riccietti, Elisa and Gribonval, R{\'e}mi},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
