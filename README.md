# py-distance-transforms

| Docs | Description |
|------|-------------|
| Getting Started| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YNcN0sTk4pu1f79KZLK9dnd4BBqKGqSv?usp=sharing) A quickstart guide to using `py_distance_transforms` for efficient distance transform operations on arrays. |
| Deep Learning (Hausdorff Loss)| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YNou2N6cywlosHSuBP1Yjj6RLUl-SfLV?usp=sharing) A [MONAI](https://github.com/Project-MONAI/tutorials) tutorial adjusted to show how to use the Hausdorff loss function and the corresponding `py_distance_transforms`|

`py_distance_transforms` is a Python package that provides efficient distance transform operations on arrays. It is a wrapper around the Julia package [DistanceTransforms.jl](https://github.com/Dale-Black/DistanceTransforms.jl), bringing its high-performance capabilities to the Python ecosystem.

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
from py_distance_transforms import transform_2d
import numpy as np

arr = np.random.choice([0, 1], size=(10, 10)).astype(np.float32)
result = transform_2d(arr)
```

## GPU Acceleration

```python
import torch
from py_distance_transforms import transform_gpu_2d

x_gpu = torch.rand((100, 100), device='cuda')
x_gpu = (x_gpu

`py_distance_transforms` is a Python package that provides efficient distance transform operations on arrays. It is a wrapper around the Julia package [DistanceTransforms.jl](https://github.com/Dale-Black/DistanceTransforms.jl), bringing its high-performance capabilities to the Python ecosystem.

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
from py_distance_transforms import transform_2d
import numpy as np

arr = np.random.choice([0, 1], size=(10, 10)).astype(np.float32)
result = transform_2d(arr)
```

## GPU Acceleration

```python
import torch
from py_distance_transforms import transform_gpu_2d

x_gpu = torch.rand((100, 100), device='cuda')
x_gpu = (x_gpu > 0.5).float()

gpu_transformed = transform_gpu_2d(x_gpu)
```

## Benchmarks

`py_distance_transforms` offers significant performance improvements compared to other Python libraries, especially when utilizing multi-threading and GPU acceleration. Detailed benchmarks can be found in the [documentation](link-to-docs).

## Documentation

For more detailed information on the usage and capabilities of `py_distance_transforms`, please refer to the [documentation](link-to-docs).

## Contributing

Contributions are welcome! Please see the [contributing guidelines](link-to-contributing-guidelines) for more information.

## License

`py_distance_transforms` is released under the [MIT License](link-to-license).

## Acknowledgments

`py_distance_transforms` is a Python wrapper around the Julia package [DistanceTransforms.jl](https://github.com/MolloiLab/DistanceTransforms.jl). We extend our gratitude to the developers and contributors of DistanceTransforms.jl for their excellent work.
