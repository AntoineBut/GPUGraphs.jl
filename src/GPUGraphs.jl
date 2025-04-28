module GPUGraphs

using KernelAbstractions
using Atomix
using GPUArrays
using Graphs
using SparseArrays
import SparseArrays: sprand
using LinearAlgebra



include("algebra.jl")
include("GPUGraphsMatrix.jl")
include("GPUGraphsVector.jl")
include("spmv.jl")
include("spmspv.jl")
include("e_wise_ops.jl")
include("map_and_reduce.jl")
include("algorithms/bfs.jl")


export AbstractSparseGPUMatrix,
    SparseGPUMatrixCSR,
    SparseGPUMatrixCSC,
    SparseGPUMatrixELL,
    SparseGPUVector,
    AbstractSparseGPUVector,
    gpu_spmv!,
    sprand_gpu,
    monoid_neutral,
    bfs_distances,
    bfs_distances!,
    GPUGraphs_add,
    GPUGraphs_mul,
    GPUGraphs_second

end
