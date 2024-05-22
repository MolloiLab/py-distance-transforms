import numpy as np
import torch
from .julia_import import jl

# Import DLPack in Julia
jl.seval("using DLPack")

def transform_1d(arr):
    if isinstance(arr, np.ndarray):
        arr_jl = jl.convert(jl.Array, arr)
        result_jl = jl.transform(jl.boolean_indicator(arr_jl))
        return np.asarray(result_jl, dtype=arr.dtype)
    else:
        raise TypeError(
            "Input must be a NumPy array. For GPU tensors, use transform_gpu_1d."
        )

def transform_2d(arr, threaded=True):
    if isinstance(arr, np.ndarray):
        arr_jl = jl.convert(jl.Array, arr)
        result_jl = jl.transform(jl.boolean_indicator(arr_jl), threaded=threaded)
        return np.asarray(result_jl, dtype=arr.dtype)
    else:
        raise TypeError(
            "Input must be a NumPy array. For GPU tensors, use transform_gpu_2d."
        )

def transform_3d(arr, threaded=True):
    if isinstance(arr, np.ndarray):
        arr_jl = jl.convert(jl.Array, arr)
        result_jl = jl.transform(jl.boolean_indicator(arr_jl), threaded=threaded)
        return np.asarray(result_jl, dtype=arr.dtype)
    else:
        raise TypeError(
            "Input must be a NumPy array. For GPU tensors, use transform_gpu_3d."
        )

def transform_gpu_1d(arr):
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda or arr.is_hip:
            # Share the PyTorch GPU tensor with Julia using DLPack
            gpu_arr = jl.from_dlpack(arr)
            result_jl = jl.transform(jl.boolean_indicator(gpu_arr))
            # Share the result back to Python using DLPack
            result_torch = torch.utils.dlpack.from_dlpack(jl.DLPack.share(result_jl, torch.utils.dlpack.to_dlpack))
            return result_torch
        else:
            raise ValueError("Input must be a GPU tensor.")
    else:
        raise TypeError("Input must be a PyTorch GPU tensor.")

def transform_gpu_2d(arr):
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda or arr.is_hip:
            # Share the PyTorch GPU tensor with Julia using DLPack
            gpu_arr = jl.from_dlpack(arr)
            result_jl = jl.transform(jl.boolean_indicator(gpu_arr))
            # Share the result back to Python using DLPack
            result_torch = torch.utils.dlpack.from_dlpack(jl.DLPack.share(result_jl, torch.utils.dlpack.to_dlpack))
            return result_torch
        else:
            raise ValueError("Input must be a GPU tensor.")
    else:
        raise TypeError("Input must be a PyTorch GPU tensor.")

def transform_gpu_3d(arr):
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda or arr.is_hip:
            # Share the PyTorch GPU tensor with Julia using DLPack
            gpu_arr = jl.from_dlpack(arr)
            result_jl = jl.transform(jl.boolean_indicator(gpu_arr))
            # Share the result back to Python using DLPack
            result_torch = torch.utils.dlpack.from_dlpack(jl.DLPack.share(result_jl, torch.utils.dlpack.to_dlpack))
            return result_torch
        else:
            raise ValueError("Input must be a GPU tensor.")
    else:
        raise TypeError("Input must be a PyTorch GPU tensor.")