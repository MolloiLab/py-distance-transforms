import juliacall as jc
from juliacall import Main as jl

jl.seval("using Pkg; Pkg.status()")
jl.seval("using PythonCall")
jl.seval("using DLPack")
jl.seval("Using CUDA")
jl.seval("using DistanceTransforms")

PythonCall = jl.PythonCall
DLPack = jl.DLPack
DistanceTransforms = jl.DistanceTransforms
