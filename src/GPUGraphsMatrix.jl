# Sparse GPU-Compatible Matrix Type and Operations for GraphBLAS in Julia
# Some parts of this code were heavily inspired from SparseArrays.jl and SuiteSparseGraphBLAS.jl
"
    Abstract type for a GPU compatible sparse matrix
"
abstract type AbstractSparseGPUMatrix{Tv,Ti<:Integer} <:
              SparseArrays.AbstractSparseMatrix{Tv,Ti} end
"""
    SparseGPUMatrixCSR{Tv,Ti<:Integer} <: AbstractSparseMatrixCSC{Tv,Ti}
    Tv : Type of the stored values
    Ti : Type of the stored indices
    A sparse matrix in Compressed Sparse Row format for GPU graph processing.
    
    The matrix is stored in CSR format, with the following fields:
    - `m::Int` : Number of rows
    - `n::Int` : Number of columns
    - `rowptr::AbstractGPUVector{Ti}` : Row i is in rowptr[i]:(rowptr[i+1]-1)
    - `colval::AbstractGPUVector{Ti}` : Col indices of stored values
    - `nzval::AbstractGPUVector{Tv}` : Stored values, typically nonzeros
    - `backend::Backend` : Backend, e.g. CPU or GPU
"""
mutable struct SparseGPUMatrixCSR{
    Tv,
    Ti<:Integer,
    Gv<:AbstractVector{Tv}, # Cannot put AbstractGPUVector here because KA's CPU backend uses Vector, 
    Gi<:AbstractVector{Ti}, # and we want to be able to use the CPU backend for testing
    B<:KernelAbstractions.Backend,
} <: AbstractSparseGPUMatrix{Tv,Ti}
    m::Int
    n::Int
    rowptr::Gi
    colval::Gi
    nzval::Gv
    backend::B
    """
        Constructors for a SparseGPUMatrixCSR
        SparseGPUMatrixCSR(m::Int, n::Int, rowptr::AbstractGPUVector{Ti}, colval::AbstractGPUVector{Ti}, nzval::AbstractGPUVector{Tv}, backend::Backend)
        SparseGPUMatrixCSR(m::Transpose{Tv,<:SparseMatrixCSC}, backend::Backend)
        SparseGPUMatrixCSR(m::Matrix{Tv}, backend::Backend)
        SparseGPUMatrixCSR(m::SparseMatrixCSC{Tv}, backend::Backend)
        SparseGPUMatrixCSR(::Type{Tv}, ::Type{Ti}, backend::Backend)
        SparseGPUMatrixCSR(m::Int, n::Int, ::Type{Tv}, ::Type{Ti}, backend::Backend)
    """
    function SparseGPUMatrixCSR(
        m::Int,
        n::Int,
        rowptr::Gi,
        colval::Gi,
        nzval::Gv,
        backend::B,
    ) where {
        Tv,
        Ti,
        Gv<:AbstractVector{Tv},
        Gi<:AbstractVector{Ti},
        B<:KernelAbstractions.Backend,
    }
        if length(rowptr) != m + 1
            throw(ArgumentError("length(rowptr) must be equal to m + 1"))
        end
        if length(colval) != length(nzval)
            throw(ArgumentError("length(colval) must be equal to length(nzval)"))
        end
        if get_backend(rowptr) != backend
            rowptr_gpu = allocate(backend, Ti, length(rowptr))
            copyto!(rowptr_gpu, rowptr)
        else
            rowptr_gpu = rowptr

        end

        if get_backend(colval) != backend
            colval_gpu = allocate(backend, Ti, length(colval))
            copyto!(colval_gpu, colval)
        else
            colval_gpu = colval
        end
        if get_backend(nzval) != backend
            nzval_gpu = allocate(backend, Tv, length(nzval))
            copyto!(nzval_gpu, nzval)
        else
            nzval_gpu = nzval
        end
        new{Tv,Ti,typeof(nzval_gpu),typeof(rowptr_gpu),B}(
            m,
            n,
            rowptr_gpu,
            colval_gpu,
            nzval_gpu,
            backend,
        )
    end
end


function SparseGPUMatrixCSR(m::Transpose{Tv,<:SparseMatrixCSC}, backend::Backend) where {Tv}
    m_t = m.parent
    rowptr = m_t.colptr
    colval = m_t.rowval
    nzval = m_t.nzval
    SparseGPUMatrixCSR(size(m_t, 2), size(m_t, 1), rowptr, colval, nzval, backend)
end

function SparseGPUMatrixCSR(
    m::Matrix{Tv},
    backend::Backend,
    ::Type{Ti} = Int32,
) where {Tv,Ti<:Integer}
    sparse_matrix_csc_t = transpose(convert(SparseMatrixCSC{Tv,Ti}, sparse(transpose(m)))) # Transpose to get the CSR format. TODO : make more efficient 
    SparseGPUMatrixCSR(sparse_matrix_csc_t, backend)
end

function SparseGPUMatrixCSR(m::SparseMatrixCSC{Tv}, backend::Backend) where {Tv}
    SparseGPUMatrixCSR(transpose(sparse(transpose(m))), backend)
end

# Empty constructors
function SparseGPUMatrixCSR(::Type{Tv}, ::Type{Ti}, backend::Backend) where {Tv,Ti}
    SparseGPUMatrixCSR(0, 0, Tv, Ti, backend)
end

function SparseGPUMatrixCSR(
    m::Int,
    n::Int,
    ::Type{Tv},
    ::Type{Ti},
    backend::Backend,
) where {Tv,Ti}
    rowptr = allocate(backend, Ti, m + 1)
    colval = allocate(backend, Ti, 0)
    nzval = allocate(backend, Tv, 0)
    SparseGPUMatrixCSR(m, n, rowptr, colval, nzval, backend)
end

#TODO : Bolean matrix that can omit the values and only store the indices


# Base methods for the SparseGPUMatrixCSR type
Base.size(A::SparseGPUMatrixCSR) = (A.m, A.n)
Base.size(A::SparseGPUMatrixCSR, i::Int) = (i == 1) ? A.m : A.n
Base.length(A::SparseGPUMatrixCSR) = A.m * A.n
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

# SparseArrays functions
# Function to get the number of nonzeros in the matrix
SparseArrays.nnz(A::SparseGPUMatrixCSR) = length(A.nzval)

# KA functions
KernelAbstractions.get_backend(A::SparseGPUMatrixCSR) = A.backend
