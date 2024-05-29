from juliacall import Main as jl

jl.seval("using Pkg; Pkg.status()")
jl.seval("using DLPack")
jl.seval("using CUDA")
jl.seval("using DistanceTransforms")

DLPack = jl.DLPack
DistanceTransforms = jl.DistanceTransforms
