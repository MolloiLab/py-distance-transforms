import numpy as np
from .julia_import import jl, DistanceTransforms

def transform(arr):
    arr_jl = jl.convert(jl.Array, arr)
    result_jl = DistanceTransforms.transform(DistanceTransforms.boolean_indicator(arr_jl))
    return np.asarray(result_jl, dtype=arr.dtype)
    