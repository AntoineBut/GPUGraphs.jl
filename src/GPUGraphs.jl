module GPUGraphs

using KernelAbstractions
using GPUArrays
using Graphs
using SparseArrays
import SparseArrays: sprand
using LinearAlgebra

export AbstractSparseGPUMatrix,
    SparseGPUMatrixCSR,
    SparseGPUMatrixCSC,
    SparseGPUMatrixELL,
    SparseGPUVector,
    AbstractSparseGPUVector,
    gpu_spmv!,
    sprand_gpu,
    monoid_neutral

include("algebra.jl")
include("GPUGraphsMatrix.jl")
include("GPUGraphsVector.jl")
include("spmv.jl")
include("e_wise_ops.jl")
include("map_and_reduce.jl")


end
