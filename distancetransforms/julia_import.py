from juliacall import Main as jl

try:
    jl.seval("using Pkg; Pkg.status()")
    jl.seval("using DistanceTransforms")
    jl.seval("using CUDA")
except Exception as e:
    print(f"Error: {e}")
    jl.seval("using Pkg; Pkg.status()")
    raise e

DistanceTransforms = jl.DistanceTransforms
CUDA = jl.CUDA