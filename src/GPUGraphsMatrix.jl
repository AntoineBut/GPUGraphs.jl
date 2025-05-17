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
Base.show(io::IO, A::SparseGPUMatrixCSR) = println(
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

### Padded Matrix: SELL format

mutable struct SparseGPUMatrixSELL{
    Tv,
    Ti<:Integer,
    Gv<:AbstractVector{Tv}, # Cannot put AbstractGPUVector here because KA's CPU backend uses Vector, 
    Gi<:AbstractVector{Ti}, # and we want to be able to use the CPU backend for testing
} <: AbstractSparseGPUMatrix{Tv,Ti}
    m::Int
    n::Int
    slice_size::Int
    nslices::Int    # Number of slices
    nnz::Int        # Number of nonzeros
    n_stored::Int   # Number of stored values (padded)
    slice_ptr::Gi   # Index of the first element of each slice
    colval::Gi
    nzval::Gv
    """
        Constructors for a SparseGPUMatrixELL
        
    """
    function SparseGPUMatrixSELL(
        m::Int,
        n::Int,
        slice_size::Int,
        nslices::Int,
        nnz::Int,
        n_stored::Int,
        slice_ptr::Gi,
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
        if length(slice_ptr) != nslices+1
            throw(ArgumentError("length(slice_ptr) must be equal to nslices+1"))
        end
        if length(colval) != length(nzval) || length(colval) != n_stored
            throw(ArgumentError("length(colval) and length(nzval) must be equal to n_stored"))
        end
        if !isempty(colval) && (maximum(colval) > n || minimum(colval) < 0)
            throw(ArgumentError("colval contains an index out of bounds"))
        end

        if get_backend(slice_ptr) != backend
            slice_ptr_gpu = allocate(backend, Ti, length(slice_ptr))
            copyto!(slice_ptr_gpu, slice_ptr)
        else
            slice_ptr_gpu = slice_ptr
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
        new{Tv,Ti,typeof(nzval_gpu),typeof(slice_ptr_gpu)}(
            m,
            n,
            slice_size,
            nslices,
            nnz,
            n_stored,
            slice_ptr_gpu,
            colval_gpu,
            nzval_gpu,
        )
    end
end

function SparseGPUMatrixSELL(
    m::Matrix{Tv},
    backend::Backend,
    slice_size::Int = 32,

    ::Type{Ti} = Int32,
) where {Tv,Ti<:Integer}
    sparse_matrix_csc = convert(SparseMatrixCSC{Tv,Ti}, sparse(m))
    SparseGPUMatrixSELL(sparse_matrix_csc, slice_size, backend)
end

function SparseGPUMatrixSELL(
    m::Transpose{Tv,<:SparseMatrixCSC{Tv,Ti}},
    slice_size::Int,
    backend::Backend,
) where {Tv,Ti<:Integer}
    m_t = m.parent
    rowptr = m_t.colptr
    colval = m_t.rowval
    nzval = m_t.nzval
    
    n_slices = ceil(Int, size(m_t, 2) / slice_size)
    max_nnz_per_slice = zeros(Int, n_slices)
    nnz_per_row = diff(rowptr)

    # Compute the maximum number of nonzeros per row for each slice
    n_stored = 0
    for i = 1:n_slices
        row_start = (i-1) * slice_size + 1
        row_end = min(i * slice_size, size(m_t, 2))
        max_nnz_per_slice[i] = maximum(nnz_per_row[row_start:row_end])
        n_stored += max_nnz_per_slice[i] * slice_size
    end
    colval_padded = ones(Ti, n_stored)
    nzval_padded = zeros(Tv, n_stored)
    slice_ptr = zeros(Ti, n_slices + 1)
    slice_ptr[1] = 1
    for i = 1:n_slices
        slice_ptr[i + 1] = slice_ptr[i] + max_nnz_per_slice[i] * slice_size
    end

    for slice = 1:n_slices
        slice_start = (slice - 1) * slice_size + 1
        slice_end = min(slice * slice_size, size(m_t, 2))
        # Fill the padded sub-matrix for each slice in Row-Major order
        max_nnz = max_nnz_per_slice[slice]
        temp_colval = ones(Ti, slice_size, max_nnz)
        temp_nzval = zeros(Tv, slice_size, max_nnz)
        for row in slice_start:slice_end
            if row > size(m_t, 2)
                break
            end
            start = rowptr[row]
            end_ = rowptr[row + 1] - 1
            temp_colval[row - slice_start + 1, 1:(end_ - start + 1)] = colval[start:end_]
            temp_nzval[row - slice_start + 1, 1:(end_ - start + 1)] = nzval[start:end_]
        end
        # Reshape the sub-matrix to make it column-major vector and copy it to final storage
       
        colval_padded[slice_ptr[slice]:slice_ptr[slice + 1] - 1] =
            collect(Iterators.flatten(temp_colval)) # matrix -> transposed matrix -> vector with inverted dimensions
        nzval_padded[slice_ptr[slice]:slice_ptr[slice + 1] - 1] =
            collect(Iterators.flatten(temp_nzval))


    end

    SparseGPUMatrixSELL(
        size(m_t, 2),
        size(m_t, 1),
        slice_size,
        n_slices,
        nnz(m_t),
        n_stored,
        slice_ptr,
        colval_padded,
        nzval_padded,
        backend,
    )

end

function SparseGPUMatrixSELL(
    m::SparseMatrixCSC{Tv,Ti},
    slice_size::Int,
    backend::Backend,
) where {Tv,Ti<:Integer}
    SparseGPUMatrixSELL(transpose(sparse(transpose(m))), slice_size, backend)
end


# Base methods for the SparseGPUMatrixCSR type
Base.size(A::SparseGPUMatrixSELL) = (A.m, A.n)
Base.size(A::SparseGPUMatrixSELL, i::Int) = (i == 1) ? A.m : A.n
Base.length(A::SparseGPUMatrixSELL) = A.m * A.n
Base.show(io::IO, A::SparseGPUMatrixSELL) = println(
    io,
    "SparseGPUMatrixSELL{$(eltype(A.nzval)) - $(eltype(A.colval))}($(size(A, 1)), $(size(A, 2))) - $(nnz(A)) explicit elements",
)
Base.display(A::SparseGPUMatrixSELL) = show(stdout, A)



function Base.getindex(A::SparseGPUMatrixSELL, i::Int, j::Int)
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

function Base.setindex!(A::SparseGPUMatrixSELL, v, i::Int, j::Int)
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
SparseArrays.nnz(A::SparseGPUMatrixSELL) = A.nnz
#sprand_gpu(::Type{Tv}, m::Int, n::Int, p::Real, backend::Backend) where {Tv} =
#    SparseGPUMatrixCSR(transpose(SparseArrays.sprand(Tv, m, n, p)), backend)

# KA functions
KernelAbstractions.get_backend(A::SparseGPUMatrixSELL) = get_backend(A.nzval)


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
    row = findfirst(A.rowval[A.colptr[j]:A.colptr[j+1]-1] .== i)
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
    row = findfirst(A.rowval[A.colptr[j]:A.colptr[j+1]-1] .== i)
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
