import torch
from .julia_import import jc, PythonCall, DLPack, DistanceTransforms

def transform_cuda(tensor):
    tensor_jl = DLPack.from_dlpack(jc.convert(PythonCall.Py, tensor))
    result_jl = DistanceTransforms.transform(DistanceTransforms.boolean_indicator(tensor_jl))
    return DLPack.share(result_jl, torch.from_dlpack)
    