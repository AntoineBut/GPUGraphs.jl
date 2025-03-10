# Sparse Vector Type and Operations for GraphBLAS in Julia

# Sparse GPU-Compatible Matrix Type and Operations for GraphBLAS in Julia
# Some parts of this code were heavily inspired from SparseArrays.jl and SuiteSparseGraphBLAS.jl

using Base
using KernelAbstractions
using GPUArrays
using SparseArrays
using LinearAlgebra

export SparseGPUMatrixCSR

# Abstract type for a GPU compatible sparse matrix
abstract type AbstractSparseGPUVector{Tv,Ti<:Integer} <:
              SparseArrays.AbstractSparseVector{Tv,Ti} end
"""
    GPUGraphsMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    Tv : Type of the stored values
    Ti : Type of the stored indices
    A sparse matrix in Compressed Sparse Row format.
"""
mutable struct SparseGPUVector{Tv,Ti<:Integer} <: AbstractSparseGPUVector{Tv,Ti}
    n::Int                  # Size
    nzind::AbstractGPUVector{Ti}      # index of stored values
    nzval::AbstractGPUVector{Tv}    # Stored values, typically nonzeros
    backend::KernelAbstractions.Backend
end

function SparseGPUVector(
    n::Int,
    nzind::AbstractGPUVector{Ti},
    nzval::AbstractGPUVector{Tv},
) where {Tv,Ti<:Integer}
    if length(nzind) != length(nzval)
        throw(ArgumentError("length(nzind) must be equal to length(nzval)"))
    end
    if !(get_backend(nzind) === get_backend(nzval))
        throw(ArgumentError("All vectors must be on the same backend"))
    else
        backend = get_backend(nzind)
    end
    SparseGPUVector(n, nzind, nzval, backend)
end

function SparseGPUVector(
    n::Int,
    nzind::Vector{Ti},
    nzval::Vector{Tv},
    backend::Backend,
) where {Tv,Ti<:Integer}
    nzind_gpu = allocate(backend, Ti, length(nzind))
    copyto!(nzind_gpu, nzind)
    nzval_gpu = allocate(backend, Tv, length(nzval))
    copyto!(nzval_gpu, nzval)
    SparseGPUVector(n, nzind_gpu, nzval_gpu, backend)
end

function SparseGPUVector(v::Vector{Tv}, backend::Backend) where {Tv}
    nzind = findall(!iszero, v)
    nzval = v[nzind]
    SparseGPUVector(length(v), nzind, nzval, backend)
end


function SparseGPUVector(v::SparseVector{Tv,Ti}, backend::Backend) where {Tv,Ti<:Integer}
    SparseGPUVector(length(v), v.nzind, v.nzval, backend)
end

# Empty constructors
function SparseGPUVector(Tv::Type, Ti::Type, backend)
    SparseGPUVector(0, Tv, Ti, backend)
end

function SparseGPUVector(n::Int, Tv::Type, Ti::Type, backend::Backend)
    nzind = allocate(backend, Ti, 0)
    nzval = allocate(backend, Tv, 0)
    SparseGPUVector(n, nzind, nzval, backend)
end

#TODO : Bolean matrix that can omit the values and only store the indices


# Basic methods for the SparseGPUMatrixCSR type

Base.size(V::SparseGPUVector) = V.n
Base.size(V::SparseGPUVector, i::Int) = (i == 1) ? V.n : 1
# Function to get the number of nonzeros in the matrix

SparseArrays.nnz(V::SparseGPUVector) = length(V.nzval)


function Base.getindex(V::SparseGPUVector, i::Int)
    @warn "Scalar indexing on a SparseGPUVector is slow. For better performance, vectorize the operation."
    if i < 1 || i > V.n
        throw(BoundsError(V, i))
    end
    pos = findfirst(V.nzind .== i)
    if pos === nothing
        return zero(eltype(V.nzval))
    else
        return V.nzval[pos]
    end
end

function Base.setindex!(V::SparseGPUVector, val, i::Int)
    @warn "Scalar indexing on a SparseGPUVector is slow. For better performance, vectorize the operation."
    if i < 1 || i > V.n
        throw(BoundsError(V, i))
    end
    pos = findfirst(V.nzind .== i)
    if pos === nothing
        throw(
            ArgumentError(
                "Index ($i) is not in the vector. Adding new values is not supported yet.",
            ),
        ) # TODO : Implement adding new values
    else
        V.nzval[pos] = val
    end
end
