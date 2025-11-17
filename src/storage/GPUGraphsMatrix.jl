# Sparse GPU-Compatible Matrix Type and Operations for GraphBLAS in Julia
# Some parts of this code were heavily inspired from SparseArrays.jl and SuiteSparseGraphBLAS.jl
"
    Abstract type for a GPU compatible sparse matrix
"
abstract type AbstractSparseGPUMatrix{Tv,Ti<:Integer} <:
              SparseArrays.AbstractSparseMatrix{Tv,Ti} end

include("CSC.jl")
include("CSR.jl")
include("SELL.jl")
