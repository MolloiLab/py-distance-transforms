import numpy as np
import torch
from .julia_import import jl

def transform_1d(arr):
    if isinstance(arr, np.ndarray):
        arr = jl.convert(jl.Array, arr)
    elif isinstance(arr, torch.Tensor):
        arr = jl.convert(jl.Array, arr.numpy())
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    
    return jl.transform(jl.boolean_indicator(arr))

def transform_2d(arr, threaded=True):
    if isinstance(arr, np.ndarray):
        arr = jl.convert(jl.Array, arr)
    elif isinstance(arr, torch.Tensor):
        arr = jl.convert(jl.Array, arr.numpy())
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    
    return jl.transform(jl.boolean_indicator(arr), threaded=threaded)

def transform_3d(arr, threaded=True):
    if isinstance(arr, np.ndarray):
        arr = jl.convert(jl.Array, arr)
    elif isinstance(arr, torch.Tensor):
        arr = jl.convert(jl.Array, arr.numpy())
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")
    
    return jl.transform(jl.boolean_indicator(arr), threaded=threaded)

def transform_gpu_2d(arr):
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            sz = tuple(arr.size())
            ptr = arr.data_ptr()
            cu_ptr = jl.seval("""
            ptr -> CuPtr{Float32}(pyconvert(UInt, ptr))
            """)(ptr)
            cu_arr = jl.unsafe_wrap(jl.CuArray, cu_ptr, sz)
            return jl.transform(jl.boolean_indicator(cu_arr))
        else:
            raise ValueError("Input must be a CUDA tensor.")
    else:
        raise TypeError("Input must be a PyTorch CUDA tensor.")

def transform_gpu_3d(arr):
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            sz = tuple(arr.size())
            ptr = arr.data_ptr()
            cu_ptr = jl.seval("""
            ptr -> CuPtr{Float32}(pyconvert(UInt, ptr))
            """)(ptr)
            cu_arr = jl.unsafe_wrap(jl.CuArray, cu_ptr, sz)
            return jl.transform(jl.boolean_indicator(cu_arr))
        else:
            raise ValueError("Input must be a CUDA tensor.")
    else:
        raise TypeError("Input must be a PyTorch CUDA tensor.")