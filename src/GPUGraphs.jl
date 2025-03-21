module GPUGraphs

using KernelAbstractions
using GPUArrays
using Graphs
using SparseArrays
using LinearAlgebra

export SparseGPUMatrixCSR,
    AbstractSparseGPUMatrix,
    SparseGPUVector,
    AbstractSparseGPUVector,
    Semiring,
    Monoid,
    GPU_spmul!


include("algebra.jl")
include("GPUGraphsMatrix.jl")
include("GPUGraphsVector.jl")
include("graphBLAS.jl")


end
