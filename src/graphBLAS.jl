# This files contains implementations of GraphBLAS operations for sparse matrices and vectors.

# Priority : efficient elementwise operations using mapreduce, Matrix-Vector products, Matrix-Matrix products with GraphBLAS semirings

@kernel function csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C using the semiring semiring.
    row = @index(Global, Linear)
    acc = monoid_neutral(eltype(a_nz_val), add)
    for i = a_row_ptr[row]:a_row_ptr[row+1]-1
        acc = add(acc, mul(b[a_col_val[i]], a_nz_val[i]))
    end
    c[row] = accum(c[row], acc)
end

function gpu_spmv!(
    C::AV,
    A::SparseGPUMatrixCSR{Tv,Ti},
    B::AV,
    mul::Function = *,
    add::Function = +,
    accum::Function = +,
) where {Tv,Ti,AV<:AbstractVector{Tv}}
    # Computes A*B and stores the result in C using the semiring semiring.
    # Check dimensions
    if size(A, 2) != length(B)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    if size(C, 1) != size(A, 1)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    # Call the kernel
    backend = get_backend(C)
    kernel! = csr_spmv_kernel!(backend)
    kernel!(C, A.rowptr, A.colval, A.nzval, B, mul, add, accum; ndrange = size(A, 1))
end

@kernel function ell_spmv_kernel!(
    c,
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(a_nnz_per_row),
    @Const(n),
    @Const(b),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C using the semiring semiring.
    row = @index(Global, Linear)
    acc = monoid_neutral(eltype(a_nz_val), add)
    for idx = row:n:a_nnz_per_row[row]*n+row-1
        acc = add(acc, mul(b[a_col_val[idx]], a_nz_val[idx]))
    end
    c[row] = accum(c[row], acc)
end

function gpu_spmv!(
    C::AV,
    A::SparseGPUMatrixELL{Tv,Ti},
    B::AV,
    mul::Function = *,
    add::Function = +,
    accum::Function = +,
) where {Tv,Ti,AV<:AbstractVector{Tv}}
    # Computes A*B and stores the result in C using the semiring semiring.
    # Check dimensions
    if size(A, 2) != length(B)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    if size(C, 1) != size(A, 1)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    # Call the kernel
    backend = get_backend(A)
    kernel! = ell_spmv_kernel!(backend)
    kernel!(
        C,
        A.colval,
        A.nzval,
        A.nnz_per_row,
        A.n,
        B,
        mul,
        add,
        accum;
        ndrange = size(A, 1),
    )
end


"""
function gpu_spmv!(
    C::AV,
    A::SparseGPUMatrixELL{Tv,Ti},
    B::AV,
    GROUP_SIZE::Int,
    mul::Function = *,
    add::Function = +,
    accum::Function = +,
    ) where {Tv,Ti,AV<:AbstractVector{Tv}}
    # Computes A*B and stores the result in C using the semiring semiring.
    
    # Check dimensions
    if size(A, 2) != length(B)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    if size(C, 1) != size(A, 1)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    
    # Call the kernel
    max_nnz_per_row = maximum(A.nnz_per_row)
    nthreads = size(A, 1) ÷ GROUP_SIZE + 1
    backend = get_backend(A)
    kernel! = ell_spmv_kernel_grouped!(backend)
    kernel!(
        C,
        A.colval,
        A.nzval,
        A.nnz_per_row,
        A.n,
        max_nnz_per_row,
        Int32(GROUP_SIZE),
        B,
        mul,
        add,
        accum;
        ndrange = nthreads,
        )
        #end
        #KernelAbstractions.synchronize(backend)
    end
    
    
    @kernel function ell_spmv_kernel_grouped!(
        c,
        @Const(a_col_val),
        @Const(a_nz_val),
        @Const(a_nnz_per_row),
        @Const(n),
        @Const(max_nnz_per_row),
        @Const(group_size),
        b,
        mul,
        add,
        accum,
    )
        # Computes A*B and stores the result in C using the semiring semiring.
        group_start = (@index(Global, Linear) - 1) * group_size + 1
        #accs = KernelAbstractions.zeros(get_backend(a_nz_val), monoid_neutral(eltype(a_nz_val), add), group_size)
        # Grouped ELL kernel
        
    
        for row = group_start:group_start+group_size-1
            if row > length(a_nnz_per_row)
                break
            end
            acc = monoid_neutral(eltype(a_nz_val), add)
            for iter = 0:max_nnz_per_row-1
                idx = row + iter * n
                acc = add(acc, mul(b[a_col_val[idx]], a_nz_val[idx]))
            end
            c[row] = accum(c[row], acc)
        end
    end
    """
