from juliacall import Main as jl

jl.seval("using Pkg; Pkg.status()")
jl.seval("using DistanceTransforms")
jl.seval("using DLPack")

DistanceTransforms = jl.DistanceTransforms
DLPack = jl.DLPack
