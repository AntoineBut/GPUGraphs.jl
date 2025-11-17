### Padded Matrix: SELL format

mutable struct SparseGPUMatrixSELL{
    Tv,
    Ti<:Integer,
    Gv<:AbstractVector{Tv}, # Cannot put AbstractGPUVector here because KA's CPU backend uses Vector, 
    Gi<:AbstractVector{Ti}, # and we want to be able to use the CPU backend for testing
} <: AbstractSparseGPUMatrix{Tv,Ti}
    m::Int
    n::Int
    perm::Vector{Int}  # Row permutation to reduce padding (not implemented yet)
    slice_size::Int
    nslices::Int    # Number of slices
    nnz::Int        # Number of nonzeros
    n_stored::Int   # Number of stored values (padded)
    slice_ptr::Gi   # Index of the first element of each slice
    colval::Gi
    nzval::Gv
    """
        Constructors for a SparseGPUMatrixSELL
        
    """
    function SparseGPUMatrixSELL(
        m::Int,
        n::Int,
        perm::Vector{Int},
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
            throw(
                ArgumentError("length(colval) and length(nzval) must be equal to n_stored"),
            )
        end
        if !isempty(colval) && (maximum(colval) > n || minimum(colval) < -1)
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
            perm,
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
    SparseGPUMatrixSELL(sparse_matrix_csc, backend, slice_size)
end

function SparseGPUMatrixSELL(
    m::Transpose{Tv,<:SparseMatrixCSC{Tv,Ti}},
    backend::Backend,
    slice_size::Int = 32,
) where {Tv,Ti<:Integer}
    m_t = m.parent
    rowptr = m_t.colptr
    colval = m_t.rowval
    nzval = m_t.nzval
    slice_size = min(slice_size, size(m_t, 2))
    n_slices = ceil(Int, size(m_t, 2) / slice_size)
    max_nnz_per_slice = zeros(Int, n_slices)
    nnz_per_row = diff(rowptr)

    # Compute optimal permutation of rows to minimize padding (not implemented yet)
    #perm = reverse!(sortperm(nnz_per_row[:]))
    perm = collect(1:size(m_t, 2))
    nnz_per_row = nnz_per_row[perm]

    # Compute the maximum number of nonzeros per row for each slice
    n_stored = 0
    for i = 1:n_slices
        row_start = (i-1) * slice_size + 1
        row_end = min(i * slice_size, size(m_t, 2))
        max_nnz_per_slice[i] = maximum(nnz_per_row[row_start:row_end])
        n_stored += max_nnz_per_slice[i] * slice_size
    end
    colval_padded = zeros(Ti, n_stored)
    nzval_padded = zeros(Tv, n_stored)
    slice_ptr = zeros(Ti, n_slices + 1)
    slice_ptr[1] = 1
    for i = 1:n_slices
        slice_ptr[i+1] = slice_ptr[i] + max_nnz_per_slice[i] * slice_size
    end

    for slice = 1:n_slices
        slice_start = (slice - 1) * slice_size + 1
        slice_end = min(slice * slice_size, size(m_t, 2))
        # Fill the padded sub-matrix for each slice in Row-Major order
        max_nnz = max_nnz_per_slice[slice]

        ### Padding for vals is 0 and for col indices is -1 (invalid index)
        temp_colval = ones(Ti, slice_size, max_nnz) .* -1
        temp_nzval = zeros(Tv, slice_size, max_nnz)
        for row = slice_start:slice_end
            if row > size(m_t, 2)
                break
            end

            start = rowptr[perm[row]]
            end_ = rowptr[perm[row]+1] - 1
            temp_colval[row-slice_start+1, 1:(end_-start+1)] = colval[start:end_]
            temp_nzval[row-slice_start+1, 1:(end_-start+1)] = nzval[start:end_]


        end
        
        # Reshape the sub-matrix to make it column-major vector and copy it to final storage

        colval_padded[slice_ptr[slice]:(slice_ptr[slice+1]-1)] =
            collect(Iterators.flatten(temp_colval)) # matrix -> transposed matrix -> vector with inverted dimensions
        nzval_padded[slice_ptr[slice]:(slice_ptr[slice+1]-1)] =
            collect(Iterators.flatten(temp_nzval))



    end

    SparseGPUMatrixSELL(
        size(m_t, 2),
        size(m_t, 1),
        perm,
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
    backend::Backend,
    slice_size::Int = 32,
) where {Tv,Ti<:Integer}
    SparseGPUMatrixSELL(transpose(sparse(transpose(m))), backend, slice_size)
end


# Base methods for the SparseGPUMatrixSELL type
Base.size(A::SparseGPUMatrixSELL) = (A.m, A.n)
Base.size(A::SparseGPUMatrixSELL, i::Int) = (i == 1) ? A.m : A.n
Base.length(A::SparseGPUMatrixSELL) = A.m * A.n
Base.show(io::IO, A::SparseGPUMatrixSELL) = println(
    io,
    "SparseGPUMatrixSELL{$(eltype(A.nzval)) - $(eltype(A.colval))}($(size(A, 1)), $(size(A, 2))) - $(nnz(A)) explicit elements",
)
Base.display(A::SparseGPUMatrixSELL) = show(stdout, A)



function Base.getindex(A::SparseGPUMatrixSELL, i::Int, j::Int)
    #@warn "Scalar indexing on a SparseGPUMatrixSELL is slow. For better performance, vectorize the operation."
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

