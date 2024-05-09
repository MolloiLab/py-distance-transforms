from .julia_import import jl, DistanceTransforms

def transform(arr):
    return jl.transform(DistanceTransforms.boolean_indicator(arr))