from juliacall import Main as jl

jl.seval("using DistanceTransforms")
jl.seval("using CUDA")

DistanceTransforms = jl.DistanceTransforms
CUDA = jl.CUDA