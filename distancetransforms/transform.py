import numpy as np
import torch
from .julia_import import jl

def transform_1d(arr):
    if isinstance(arr, np.ndarray):
        arr_jl = jl.convert(jl.Array, arr)
        result_jl = jl.transform(jl.boolean_indicator(arr_jl))
        return np.asarray(result_jl, dtype=arr.dtype)
    else:
        raise TypeError("Input must be a NumPy array. For PyTorch tensors, use transform_gpu_1d.")

def transform_2d(arr, threaded=True):
    if isinstance(arr, np.ndarray):
        arr_jl = jl.convert(jl.Array, arr)
        result_jl = jl.transform(jl.boolean_indicator(arr_jl), threaded=threaded)
        return np.asarray(result_jl, dtype=arr.dtype)
    else:
        raise TypeError("Input must be a NumPy array. For PyTorch tensors, use transform_gpu_2d.")

def transform_3d(arr, threaded=True):
    if isinstance(arr, np.ndarray):
        arr_jl = jl.convert(jl.Array, arr)
        result_jl = jl.transform(jl.boolean_indicator(arr_jl), threaded=threaded)
        return np.asarray(result_jl, dtype=arr.dtype)
    else:
        raise TypeError("Input must be a NumPy array. For PyTorch tensors, use transform_gpu_3d.")

def transform_gpu_1d(arr):
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            sz = tuple(arr.size())
            ptr = arr.data_ptr()
            cu_ptr = jl.seval("""
            ptr -> CUDA.CuPtr{Float32}(pyconvert(UInt, ptr))
            """)(ptr)
            cu_arr = jl.unsafe_wrap(jl.CUDA.CuArray, cu_ptr, sz)
            result_jl = jl.transform(jl.boolean_indicator(cu_arr))
            result_np = np.asarray(result_jl)
            return torch.from_numpy(result_np).to(arr.dtype).cuda()
        else:
            raise ValueError("Input must be a CUDA tensor.")
    else:
        raise TypeError("Input must be a PyTorch CUDA tensor.")

def transform_gpu_2d(arr):
    if isinstance(arr, torch.Tensor):
        if arr.is_cuda:
            sz = tuple(arr.size())
            ptr = arr.data_ptr()
            cu_ptr = jl.seval("""
            ptr -> CUDA.CuPtr{Float32}(pyconvert(UInt, ptr))
            """)(ptr)
            cu_arr = jl.unsafe_wrap(jl.CUDA.CuArray, cu_ptr, sz)
            result_jl = jl.transform(jl.boolean_indicator(cu_arr))
            result_np = np.asarray(result_jl)
            return torch.from_numpy(result_np).to(arr.dtype).cuda()
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
            ptr -> CUDA.CuPtr{Float32}(pyconvert(UInt, ptr))
            """)(ptr)
            cu_arr = jl.unsafe_wrap(jl.CUDA.CuArray, cu_ptr, sz)
            result_jl = jl.transform(jl.boolean_indicator(cu_arr))
            result_np = np.asarray(result_jl)
            return torch.from_numpy(result_np).to(arr.dtype).cuda()
        else:
            raise ValueError("Input must be a CUDA tensor.")
    else:
        raise TypeError("Input must be a PyTorch CUDA tensor.")