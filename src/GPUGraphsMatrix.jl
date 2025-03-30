# Sparse GPU-Compatible Matrix Type and Operations for GraphBLAS in Julia
# Some parts of this code were heavily inspired from SparseArrays.jl and SuiteSparseGraphBLAS.jl
"
    Abstract type for a GPU compatible sparse matrix
"
abstract type AbstractSparseGPUMatrix{Tv,Ti<:Integer} <:
              SparseArrays.AbstractSparseMatrix{Tv,Ti} end
"""
    SparseGPUMatrixCSR{Tv,Ti<:Integer} <: SparseArrays.AbstractSparseMatrix{Tv,Ti}
    Tv : Type of the stored values
    Ti : Type of the stored indices
    A sparse matrix in Compressed Sparse Row format for GPU graph processing.
    
    The matrix is stored in CSR format, with the following fields:
    - `m::Int` : Number of rows
    - `n::Int` : Number of columns
    - `rowptr::AbstractGPUVector{Ti}` : Row i is in rowptr[i]:(rowptr[i+1]-1)
    - `colval::AbstractGPUVector{Ti}` : Col indices of stored values
    - `nzval::AbstractGPUVector{Tv}` : Stored values, typically nonzeros
"""
mutable struct SparseGPUMatrixCSR{
    Tv,
    Ti<:Integer,
    Gv<:AbstractVector{Tv}, # Cannot put AbstractGPUVector here because KA's CPU backend uses Vector, 
    Gi<:AbstractVector{Ti}, # and we want to be able to use the CPU backend for testing
} <: AbstractSparseGPUMatrix{Tv,Ti}
    m::Int
    n::Int
    rowptr::Gi
    colval::Gi
    nzval::Gv
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
        if !isempty(colval) && (maximum(colval) > n || minimum(colval) < 1)
            throw(ArgumentError("colval contains an index out of bounds"))
        end
        if @allowscalar rowptr[1] != 1 || @allowscalar rowptr[m+1] != length(colval) + 1
            throw(
                ArgumentError(
                    "rowptr[1] must be equal to 1 and rowptr[m+1] must be equal to length(colval) + 1",
                ),
            )
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
        new{Tv,Ti,typeof(nzval_gpu),typeof(rowptr_gpu)}(
            m,
            n,
            rowptr_gpu,
            colval_gpu,
            nzval_gpu,
        )
    end
end


function SparseGPUMatrixCSR(
    m::Transpose{Tv,<:SparseMatrixCSC{Tv,Ti}},
    backend::Backend,
) where {Tv,Ti<:Integer}
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
    rowptr .= 1
    colval = allocate(backend, Ti, 0)
    nzval = allocate(backend, Tv, 0)
    SparseGPUMatrixCSR(m, n, rowptr, colval, nzval, backend)
end

#TODO : Bolean matrix that can omit the values and only store the indices


# Base methods for the SparseGPUMatrixCSR type
Base.size(A::SparseGPUMatrixCSR) = (A.m, A.n)
Base.size(A::SparseGPUMatrixCSR, i::Int) = (i == 1) ? A.m : A.n
Base.length(A::SparseGPUMatrixCSR) = A.m * A.n
Base.show(io::IO, A::SparseGPUMatrixCSR) = print(
    io,
    "SparseGPUMatrixCSR{$(eltype(A.nzval))}($(size(A, 1)), $(size(A, 2))) - $(nnz(A)) explicit elements",
)
Base.display(A::SparseGPUMatrixCSR) = show(stdout, A)



function Base.getindex(A::SparseGPUMatrixCSR, i::Int, j::Int)
    #@warn "Scalar indexing on a SparseGPUMatrixCSR is slow. For better performance, vectorize the operation."
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
sprand_gpu(::Type{Tv}, m::Int, n::Int, p::Real, backend::Backend) where {Tv} =
    SparseGPUMatrixCSR(transpose(SparseArrays.sprand(Tv, m, n, p)), backend)

# KA functions
KernelAbstractions.get_backend(A::SparseGPUMatrixCSR) = get_backend(A.nzval)

### Padded CSR Matrix 


mutable struct SparseGPUMatrixELL{
    Tv,
    Ti<:Integer,
    Gv<:AbstractVector{Tv}, # Cannot put AbstractGPUVector here because KA's CPU backend uses Vector, 
    Gi<:AbstractVector{Ti}, # and we want to be able to use the CPU backend for testing
} <: AbstractSparseGPUMatrix{Tv,Ti}
    m::Int
    n::Int
    nnz_per_row::Gi
    colval::Gi
    nzval::Gv
    """
        Constructors for a SparseGPUMatrixELL
        SparseGPUMatrixELL(m::Int, n::Int, nnz_per_row::AbstractGPUVector{Ti}, colval::AbstractGPUVector{Ti}, nzval::AbstractGPUVector{Tv}, backend::Backend)
    """
    function SparseGPUMatrixELL(
        m::Int,
        n::Int,
        nnz_per_row::Gi,
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
        if length(nnz_per_row) != m
            throw(ArgumentError("length(nnz_per_row) must be equal to m"))
        end
        if length(colval) != length(nzval)
            throw(ArgumentError("length(colval) must be equal to length(nzval)"))
        end
        if !isempty(colval) && (maximum(colval) > n || minimum(colval) < 0)
            throw(ArgumentError("colval contains an index out of bounds"))
        end

        if get_backend(nnz_per_row) != backend
            nnz_per_row_gpu = allocate(backend, Ti, length(nnz_per_row))
            copyto!(nnz_per_row_gpu, nnz_per_row)
        else
            nnz_per_row_gpu = nnz_per_row
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
        new{Tv,Ti,typeof(nzval_gpu),typeof(nnz_per_row_gpu)}(
            m,
            n,
            nnz_per_row_gpu,
            colval_gpu,
            nzval_gpu,
        )
    end
end

function SparseGPUMatrixELL(
    m::Matrix{Tv},
    backend::Backend,
    ::Type{Ti} = Int32,
) where {Tv,Ti<:Integer}
    sparse_matrix_csc = convert(SparseMatrixCSC{Tv,Ti}, sparse(m))
    SparseGPUMatrixELL(sparse_matrix_csc, backend)
end

function SparseGPUMatrixELL(
    m::Transpose{Tv,<:SparseMatrixCSC{Tv,Ti}},
    backend::Backend,
) where {Tv,Ti<:Integer}
    m_t = m.parent
    rowptr = m_t.colptr
    colval = m_t.rowval
    nzval = m_t.nzval
    nnz_per_row = diff(rowptr)
    max_nnz = maximum(nnz_per_row)

    colval_padded = zeros(Ti, length(nnz_per_row) * max_nnz)
    nzval_padded = zeros(Tv, length(nnz_per_row) * max_nnz)

    for i = 1:length(nnz_per_row)
        row_start = rowptr[i]
        row_end = rowptr[i+1]
        row_nnz = row_end - row_start
        colval_padded[(i-1)*max_nnz+1:i*max_nnz] =
            [colval[row_start:row_end-1]; ones(Ti, max_nnz - row_nnz)]
        nzval_padded[(i-1)*max_nnz+1:i*max_nnz] =
            [nzval[row_start:row_end-1]; zeros(Tv, max_nnz - row_nnz)]
    end

    SparseGPUMatrixELL(
        size(m_t, 2),
        size(m_t, 1),
        nnz_per_row,
        collect(Iterators.flatten(transpose(reshape(colval_padded, Int(max_nnz), :)))), # vector -> matrix -> vector with inverted dimensions
        collect(Iterators.flatten(transpose(reshape(nzval_padded, Int(max_nnz), :)))),
        backend,
    )
end

function SparseGPUMatrixELL(
    m::SparseMatrixCSC{Tv,Ti},
    backend::Backend,
) where {Tv,Ti<:Integer}
    SparseGPUMatrixELL(transpose(sparse(transpose(m))), backend)
end


# Base methods for the SparseGPUMatrixCSR type
Base.size(A::SparseGPUMatrixELL) = (A.m, A.n)
Base.size(A::SparseGPUMatrixELL, i::Int) = (i == 1) ? A.m : A.n
Base.length(A::SparseGPUMatrixELL) = A.m * A.n
Base.show(io::IO, A::SparseGPUMatrixELL) = print(
    io,
    "SparseGPUMatrixELL{$(eltype(A.nzval))}($(size(A, 1)), $(size(A, 2))) - $(nnz(A)) explicit elements",
)
Base.display(A::SparseGPUMatrixELL) = show(stdout, A)



function Base.getindex(A::SparseGPUMatrixELL, i::Int, j::Int)
    #@warn "Scalar indexing on a SparseGPUMatrixCSR is slow. For better performance, vectorize the operation."
    if i < 1 || i > A.m || j < 1 || j > A.n
        throw(BoundsError(A, (i, j)))
    end
    row_offset = i - 1
    # The elements of the row i are stored at col+row_offset for col striding with step = n
    for col = 1:A.n:length(A.colval)
        if A.colval[col+row_offset] == j
            return A.nzval[col+row_offset]
        end
    end
    return zero(eltype(A.nzval))


end

function Base.setindex!(A::SparseGPUMatrixELL, v, i::Int, j::Int)
    if i < 1 || i > A.m || j < 1 || j > A.n
        throw(BoundsError(A, (i, j)))
    end
    row_offset = i - 1

    for col = 1:A.n:length(A.colval)
        if A.colval[col+row_offset] == j
            A.nzval[col+row_offset] = v
            return
        end
    end
    throw(
        ArgumentError(
            "Index ($i, $j) is not in the matrix. Adding new values is not supported yet.",
        ),
    ) # TODO : Implement adding new values
end

# SparseArrays functions
# Function to get the number of nonzeros in the matrix
SparseArrays.nnz(A::SparseGPUMatrixELL) = reduce(+, A.nnz_per_row)
#sprand_gpu(::Type{Tv}, m::Int, n::Int, p::Real, backend::Backend) where {Tv} =
#    SparseGPUMatrixCSR(transpose(SparseArrays.sprand(Tv, m, n, p)), backend)

# KA functions
KernelAbstractions.get_backend(A::SparseGPUMatrixELL) = get_backend(A.nzval)
