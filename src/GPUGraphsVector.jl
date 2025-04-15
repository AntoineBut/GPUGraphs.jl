# Sparse Vector Type and Operations for GraphBLAS in Julia

# Sparse GPU-Compatible Matrix Type and Operations for GraphBLAS in Julia
# Some parts of this code were heavily inspired from SparseArrays.jl and SuiteSparseGraphBLAS.jl

# Abstract type for a GPU compatible sparse matrix
abstract type AbstractSparseGPUVector{Tv,Ti<:Integer} <:
              SparseArrays.AbstractSparseVector{Tv,Ti} end


"""
    SparseGPUVector{Tv,Ti<:Integer} <: AbstractSparseVector{Tv,Ti}
    Tv : Type of the stored values
    Ti : Type of the stored indices
    A sparse vector in Compressed Sparse Row format.
"""
mutable struct SparseGPUVector{
    Tv,
    Ti<:Integer,
    Gv<:AbstractVector{Tv},
    Gi<:AbstractVector{Ti},
} <: AbstractSparseGPUVector{Tv,Ti}
    n::Int                  # Size
    nzind::Gi     # index of stored values
    nzval::Gv    # Stored values, typically nonzeros

    function SparseGPUVector(
        n::Int,
        nzind::Gi,
        nzval::Gv,
        backend::B,
    ) where {
        Tv,
        Ti<:Integer,
        Gv<:AbstractVector{Tv},
        Gi<:AbstractVector{Ti},
        B<:KernelAbstractions.Backend,
    }
        if length(nzind) != length(nzval)
            throw(ArgumentError("length(nzind) must be equal to length(nzval)"))
        end
        if get_backend(nzind) != backend
            nzind_gpu = allocate(backend, Ti, length(nzind))
            copyto!(nzind_gpu, nzind)
        else
            nzind_gpu = nzind
        end
        if get_backend(nzval) != backend
            nzval_gpu = allocate(backend, Tv, length(nzval))
            copyto!(nzval_gpu, nzval)
        else
            nzval_gpu = nzval
        end
        new{Tv,Ti,Gv,Gi}(n, nzind_gpu, nzval_gpu)
    end
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
function SparseGPUVector(::Type{Tv}, ::Type{Ti}, backend) where {Tv,Ti<:Integer}
    SparseGPUVector(0, Tv, Ti, backend)
end

function SparseGPUVector(
    n::Int,
    ::Type{Tv},
    ::Type{Ti},
    backend::Backend,
) where {Tv,Ti<:Integer}
    nzind = allocate(backend, Ti, 0)
    nzval = allocate(backend, Tv, 0)
    SparseGPUVector(n, nzind, nzval, backend)
end

#TODO : Bolean matrix that can omit the values and only store the indices

# Base methods for the SparseGPUVector type
Base.size(V::SparseGPUVector) = (V.n)
Base.size(V::SparseGPUVector, i::Int) = (i == 1) ? V.n : 1
Base.length(V::SparseGPUVector) = V.n
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
    #@warn "Scalar indexing on a SparseGPUVector is slow. For better performance, vectorize the operation."
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

# SparseArrays functions
SparseArrays.nnz(V::SparseGPUVector) = length(V.nzval)

# KA functions
KernelAbstractions.get_backend(V::SparseGPUVector) = get_backend(V.nzval)
