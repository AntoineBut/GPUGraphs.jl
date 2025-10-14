
@kernel function csc_spmspv_kernel!(
    c,
    @Const(a_col_ptr),
    @Const(a_row_val),
    @Const(a_nz_val),
    @Const(b_nzind),
    @Const(b_nz_val),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C using the semiring semiring.
    row = @index(Global, Linear)
    acc = monoid_neutral(eltype(a_nz_val), add)
    for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)
        acc = add(acc, mul(b[a_col_val[i]], a_nz_val[i]))
        #acc = add(acc, mul(a_col_val[i], a_nz_val[i]))
    end
    c[row] = accum(c[row], acc)
end

function gpu_spmspv!(
    C::SparseGPUVector{Tv,Ti},
    A::SparseGPUMatrixCSC{Tv,Ti},
    B::SparseGPUVector{Tv,Ti},
    mul::Function = *,
    add::Function = +,
    accum::Function = +,
) where {Tv,Ti}
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
    kernel! = csc_spmspv_kernel!(backend)
    kernel!(
        C,
        A.colptr,
        A.rowval,
        A.nzval,
        B.nzind,
        B.nzval,
        mul,
        add,
        accum;
        ndrange = size(A, 1),
    )
end
