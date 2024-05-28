# py-distance-transforms
`py_distance_transforms` is a Python package that provides efficient distance transform operations on arrays. It is a wrapper around the Julia package [DistanceTransforms.jl](https://github.com/Dale-Black/DistanceTransforms.jl), bringing its high-performance capabilities to the Python ecosystem.


## Documentation
| Docs | Description |
|------|-------------|
| Getting Started: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YNcN0sTk4pu1f79KZLK9dnd4BBqKGqSv?usp=sharing) | A quickstart guide to using `py_distance_transforms` for efficient distance transform operations on arrays. |
| Deep Learning (Hausdorff Loss): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YNou2N6cywlosHSuBP1Yjj6RLUl-SfLV?usp=sharing) | A [MONAI](https://github.com/Project-MONAI/tutorials) tutorial adjusted to show how to use the Hausdorff loss function and the corresponding `py_distance_transforms`|

## Features

- Fast distance transform computations on CPU and GPU
- Support for 1D, 2D, and 3D arrays
- Multi-threading for enhanced CPU performance
- GPU acceleration for NVIDIA GPUs (CUDA)
- Simple and intuitive API

## Installation

Install `py_distance_transforms` using pip:

```bash
pip install py_distance_transforms
```

## Basic Usage

```python
from py_distance_transforms import transform
import numpy as np

arr = np.random.choice([0, 1], size=(10, 10)).astype(np.float32)
result = transform(arr)
```

## GPU Acceleration

```python
import torch
from py_distance_transforms import transform_cuda

x_gpu = torch.rand((100, 100), device='cuda')
x_gpu = (x_gpu > 0.5).float()

gpu_transformed = transform_cuda(x_gpu)
```

## Acknowledgments

- `py_distance_transforms` is a Python wrapper around the Julia package [DistanceTransforms.jl](https://github.com/MolloiLab/DistanceTransforms.jl).
- Huge thanks to @pabloferz for getting DLPack.jl to work with PythonCall/juliacall and PyTorch. Massive thanks to @cjdoris and all of the contributors to PythonCall.jl as well.
