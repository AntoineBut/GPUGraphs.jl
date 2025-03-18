module GPUGraphs

using KernelAbstractions
using GPUArrays
using Graphs
using SparseArrays
using LinearAlgebra

import AcceleratedKernels as AK

export SparseGPUMatrixCSR,
    AbstractSparseGPUMatrix,
    SparseGPUVector,
    AbstractSparseGPUVector,
    Semiring,
    Monoid,
    mul!


include("algebra.jl")

include("GPUGraphsMatrix.jl")
include("GPUGraphsVector.jl")

include("graphBLAS.jl")


end
