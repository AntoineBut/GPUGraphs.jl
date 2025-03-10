# Sparse GPU-Compatible Matrix Type and Operations for GraphBLAS in Julia
# Some parts of this code were heavily inspired from SparseArrays.jl and SuiteSparseGraphBLAS.jl

using Base
using KernelAbstractions
using GPUArrays
using SparseArrays
using LinearAlgebra


export SparseGPUMatrixCSR, AbstractSparseGPUMatrix

"
    Abstract type for a GPU compatible sparse matrix
"
abstract type AbstractSparseGPUMatrix{Tv,Ti<:Integer} <:
              SparseArrays.AbstractSparseMatrix{Tv,Ti} end
"""
    GPUGraphsMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    Tv : Type of the stored values
    Ti : Type of the stored indices
    A sparse matrix in Compressed Sparse Row format.
"""
mutable struct SparseGPUMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseGPUMatrix{Tv,Ti}
    m::Int                  # Number of rows
    n::Int                  # Number of columns
    rowptr::AbstractGPUVector{Ti}     # Row i is in rowptr[i]:(rowptr[i+1]-1)
    colval::AbstractGPUVector{Ti}     # Col indices of stored values
    nzval::AbstractGPUVector{Tv}      # Stored values, typically nonzeros
    backend::KernelAbstractions.Backend
end

function SparseGPUMatrixCSR(
    m::Int,
    n::Int,
    rowptr::AbstractGPUVector{Ti},
    colval::AbstractGPUVector{Ti},
    nzval::AbstractGPUVector{Tv},
) where {Tv,Ti<:Integer}
    if length(rowptr) != m + 1
        throw(ArgumentError("length(rowptr) must be m+1"))
    end
    if length(colval) != length(nzval)
        throw(ArgumentError("length(colval) must be equal to length(nzval)"))
    end
    if !(get_backend(rowptr) === get_backend(colval) === get_backend(nzval))
        throw(ArgumentError("All vectors must be on the same backend"))
    else
        backend = get_backend(rowptr)
    end
    SparseGPUMatrixCSR(m, n, rowptr, colval, nzval, backend)
end

function SparseGPUMatrixCSR(
    m::Int,
    n::Int,
    rowptr::Vector{Ti},
    colval::Vector{Ti},
    nzval::Vector{Tv},
    backend::Backend,
) where {Tv,Ti<:Integer}
    rowptr_gpu = allocate(backend, Ti, length(rowptr))
    copyto!(rowptr_gpu, rowptr)
    colval_gpu = allocate(backend, Ti, length(colval))
    copyto!(colval_gpu, colval)
    nzval_gpu = allocate(backend, Tv, length(nzval))
    copyto!(nzval_gpu, nzval)

    SparseGPUMatrixCSR(m, n, rowptr_gpu, colval_gpu, nzval_gpu)
end

function SparseGPUMatrixCSR(m::Transpose{Tv,<:SparseMatrixCSC}, backend::Backend) where {Tv}
    m_t = m.parent
    rowptr = m_t.colptr
    colval = m_t.rowval
    nzval = m_t.nzval
    SparseGPUMatrixCSR(size(m, 1), size(m, 2), rowptr, colval, nzval, backend)
end

function SparseGPUMatrixCSR(m::Matrix{Tv}, backend::Backend) where {Tv}
    sparse_matrix_csc_t = transpose(sparse(transpose(m))) # Transpose to get the CSR format. TODO : make more efficient 
    SparseGPUMatrixCSR(sparse_matrix_csc_t, backend)
end

function SparseGPUMatrixCSR(m::SparseMatrixCSC{Tv}, backend::Backend) where {Tv}
    SparseGPUMatrixCSR(transpose(sparse(transpose(m))), backend)
end

# Empty constructors
function SparseGPUMatrixCSR(Tv::Type, Ti::Type, backend)
    SparseGPUMatrixCSR(0, 0, Tv, Ti, backend)
end

function SparseGPUMatrixCSR(m::Int, n::Int, Tv::Type, Ti::Type, backend::Backend)
    rowptr = allocate(backend, Ti, m + 1)
    colval = allocate(backend, Ti, 0)
    nzval = allocate(backend, Tv, 0)
    SparseGPUMatrixCSR(m, n, rowptr, colval, nzval, backend)
end

#TODO : Bolean matrix that can omit the values and only store the indices


# Basic methods for the SparseGPUMatrixCSR type

Base.size(A::SparseGPUMatrixCSR) = (A.m, A.n)
Base.size(A::SparseGPUMatrixCSR, i::Int) = (i == 1) ? A.m : A.n
# Function to get the number of nonzeros in the matrix
SparseArrays.nnz(A::SparseGPUMatrixCSR) = length(A.nzval)

function Base.getindex(A::SparseGPUMatrixCSR, i::Int, j::Int)
    @warn "Scalar indexing on a SparseGPUMatrixCSR is slow. For better performance, vectorize the operation."
    if i < 1 || i > A.m || j < 1 || j > A.n
        throw(BoundsError(A, (i, j)))
    end
    col = findfirst(A.colval[A.rowptr[i]:A.rowptr[i+1]-1] .== j)
    if col === nothing
        return zero(eltype(A.nzval))
    else
        return A.nzval[A.rowptr[i]+col-1]
    end
end

function Base.setindex!(A::SparseGPUMatrixCSR, v, i::Int, j::Int)
    if i < 1 || i > A.m || j < 1 || j > A.n
        throw(BoundsError(A, (i, j)))
    end
    col = findfirst(A.colval[A.rowptr[i]:A.rowptr[i+1]-1] .== j)
    if col === nothing
        throw(
            ArgumentError(
                "Index ($i, $j) is not in the matrix. Adding new values is not supported yet.",
            ),
        ) # TODO : Implement adding new values
    else
        A.nzval[A.rowptr[i]+col-1] = v
    end
end
