# PathNorm for PyTorch

Compute path-norms for supported PyTorch models in a few lines.

```python
from pathnorm import compute_path_norm

value = compute_path_norm(model, input_shape=(3, 224, 224))
```

- Supports modern ReLU networks with skip connections, convolutions, pooling, and frozen BatchNorm.
- Raises explicit errors when a model contains an unsupported module or operation.

Quickstart notebook: [quickstart.ipynb](quickstart.ipynb)

Core implementation: [pathnorm.py](pathnorm.py)

Everything else in this repository is reproduction material for the related papers and can be ignored for standard use.

## Related Papers

This public API comes from [*A path-norm toolkit for modern networks: consequences, promises and challenges*](https://arxiv.org/abs/2310.01225), later re-used in [*A Rescaling-Invariant Lipschitz Bound Based on Path-Metrics for Modern ReLU Network Parameterizations*](https://arxiv.org/pdf/2405.15006).

This repository also contains the reproduction material for these papers, in [repro/iclr24](repro/iclr24) and [repro/icml25](repro/icml25).

The ICML folder can be of separate interest to compute path-metrics, a generalization of path-norms used to derive Lipschitz bounds with respect to the weights of a ReLU network. In that folder, it is used to decide which weights to prune in a ResNet-18 trained on ImageNet. 

## Citation

If you use `pathnorm.py`, please cite the ICLR 2024 paper. If you use the path-metric or pruning experiments, please cite the ICML 2025 paper as well.

```bibtex
@inproceedings{gonon2024pathnorm,
  title={A path-norm toolkit for modern networks: consequences, promises and challenges},
  author={Gonon, Antoine and Brisebarre, Nicolas and Riccietti, Elisa and Gribonval, R{\'e}mi},
  booktitle={International Conference on Learning Representations},
  year={2024}
}

@inproceedings{gonon2025pathmetrics,
  title={A Rescaling-Invariant Lipschitz Bound Based on Path-Metrics for Modern ReLU Network Parameterizations},
  author={Gonon, Antoine and Brisebarre, Nicolas and Riccietti, Elisa and Gribonval, R{\'e}mi},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
