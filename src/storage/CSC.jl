# Not properly supported yet.
mutable struct SparseGPUMatrixCSC{
    Tv,
    Ti<:Integer,
    Gv<:AbstractVector{Tv}, # Cannot put AbstractGPUVector here because KA's CPU backend uses Vector, 
    Gi<:AbstractVector{Ti}, # and we want to be able to use the CPU backend for testing
} <: AbstractSparseGPUMatrix{Tv,Ti}
    m::Int
    n::Int
    colptr::Gi
    rowval::Gi
    nzval::Gv
    """
        Constructors for a SparseGPUMatrixCSC
        
    """
    function SparseGPUMatrixCSC(
        m::Int,
        n::Int,
        colptr::Gi,
        rowval::Gi,
        nzval::Gv,
        backend::B,
    ) where {
        Tv,
        Ti,
        Gv<:AbstractVector{Tv},
        Gi<:AbstractVector{Ti},
        B<:KernelAbstractions.Backend,
    }
        if length(colptr) != m + 1
            throw(ArgumentError("length(colptr) must be equal to m + 1"))
        end
        if length(rowval) != length(nzval)
            throw(ArgumentError("length(rowval) must be equal to length(nzval)"))
        end
        if !isempty(rowval) && (maximum(rowval) > n || minimum(rowval) < 1)
            throw(ArgumentError("rowval contains an index out of bounds"))
        end
        if @allowscalar colptr[1] != 1 || @allowscalar colptr[m+1] != length(rowval) + 1
            throw(
                ArgumentError(
                    "colptr[1] must be equal to 1 and colptr[m+1] must be equal to length(rowval) + 1",
                ),
            )
        end
        if get_backend(colptr) != backend
            colptr_gpu = allocate(backend, Ti, length(colptr))
            copyto!(colptr_gpu, colptr)
        else
            colptr_gpu = colptr
        end

        if get_backend(rowval) != backend
            rowval_gpu = allocate(backend, Ti, length(rowval))
            copyto!(rowval_gpu, rowval)
        else
            rowval_gpu = rowval

        end
        if get_backend(nzval) != backend
            nzval_gpu = allocate(backend, Tv, length(nzval))
            copyto!(nzval_gpu, nzval)
        else
            nzval_gpu = nzval
        end
        new{Tv,Ti,typeof(nzval_gpu),typeof(colptr_gpu)}(
            m,
            n,
            colptr_gpu,
            rowval_gpu,
            nzval_gpu,
        )
    end
end

function SparseGPUMatrixCSC(
    m::Matrix{Tv},
    backend::Backend,
    ::Type{Ti} = Int32,
) where {Tv,Ti<:Integer}
    sparse_matrix_csc = SparseMatrixCSC{Tv,Ti}(m)
    SparseGPUMatrixCSC(sparse_matrix_csc, backend)
end

function SparseGPUMatrixCSC(m::SparseMatrixCSC{Tv}, backend::Backend) where {Tv}
    SparseGPUMatrixCSC(size(m, 1), size(m, 2), m.colptr, m.rowval, m.nzval, backend)
end

# Empty constructors
function SparseGPUMatrixCSC(::Type{Tv}, ::Type{Ti}, backend::Backend) where {Tv,Ti}
    SparseGPUMatrixCSC(0, 0, Tv, Ti, backend)
end

function SparseGPUMatrixCSC(
    m::Int,
    n::Int,
    ::Type{Tv},
    ::Type{Ti},
    backend::Backend,
) where {Tv,Ti}
    colptr = allocate(backend, Ti, m + 1)
    colptr .= 1
    rowval = allocate(backend, Ti, 0)
    nzval = allocate(backend, Tv, 0)
    SparseGPUMatrixCSC(m, n, colptr, rowval, nzval, backend)
end

#TODO : Bolean matrix that can omit the values and only store the indices


# Base methods for the SparseGPUMatrixCSC type
Base.size(A::SparseGPUMatrixCSC) = (A.m, A.n)
Base.size(A::SparseGPUMatrixCSC, i::Int) = (i == 1) ? A.m : A.n
Base.length(A::SparseGPUMatrixCSC) = A.m * A.n
Base.show(io::IO, A::SparseGPUMatrixCSC) = println(
    io,
    "SparseGPUMatrixCSC{$(eltype(A.nzval))}($(size(A, 1)), $(size(A, 2))) - $(nnz(A)) explicit elements",
)
Base.display(A::SparseGPUMatrixCSC) = show(stdout, A)



function Base.getindex(A::SparseGPUMatrixCSC, i::Int, j::Int)
    #@warn "Scalar indexing on a SparseGPUMatrixCSC is slow. For better performance, vectorize the operation."
    if i < 1 || i > A.m || j < 1 || j > A.n
        throw(BoundsError(A, (i, j)))
    end
    row = findfirst(A.rowval[A.colptr[j]:(A.colptr[j+1]-1)] .== i)
    if row === nothing
        return zero(eltype(A.nzval))
    else
        return A.nzval[A.colptr[j]+row-1]
    end
end

function Base.setindex!(A::SparseGPUMatrixCSC, v, i::Int, j::Int)
    if i < 1 || i > A.m || j < 1 || j > A.n
        throw(BoundsError(A, (i, j)))
    end
    row = findfirst(A.rowval[A.colptr[j]:(A.colptr[j+1]-1)] .== i)
    if row === nothing
        throw(
            ArgumentError(
                "Index ($i, $j) is not in the matrix. Adding new values is not supported yet.",
            ),
        ) # TODO : Implement adding new values
    else
        A.nzval[A.colptr[j]+row-1] = v
    end
end

# SparseArrays functions
# Function to get the number of nonzeros in the matrix
SparseArrays.nnz(A::SparseGPUMatrixCSC) = length(A.nzval)

# KA functions
KernelAbstractions.get_backend(A::SparseGPUMatrixCSC) = get_backend(A.nzval)
