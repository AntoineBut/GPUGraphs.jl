# Not properly supported yet
@kernel function csc_spmv_kernel!(
    c,
    @Const(a_col_ptr),
    @Const(a_row_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(monoid_neutral_element),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    col = @index(Global, Linear)
    acc = monoid_neutral_element
    for i = a_col_ptr[col]:(a_col_ptr[col+1]-1)
        row = a_row_val[i]
        acc = mul(b[col], a_nz_val[i], row, col, col, 1)
        Atomix.@atomic c[row] += acc
    end
end

function gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixCSC{Tv,Ti},
    B::InputVec;
    mul::Function = GPUGraphs_mul,
    add::Function = GPUGraphs_add,
    accum::Function = GPUGraphs_second,
) where {
    Tv,
    Ti,
    ResType<:Number,
    InputType<:Number,
    ResVec<:AbstractVector{ResType},
    InputVec<:AbstractVector{InputType},
}
    # Computes A*B and stores the result in C
    # Check dimensions
    if size(A, 2) != length(B)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    if size(C, 1) != size(A, 1)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    # Call the kernel
    backend = get_backend(C)
    kernel! = csc_spmv_kernel!(backend)
    kernel!(
        C,
        A.colptr,
        A.rowval,
        A.nzval,
        B,
        monoid_neutral(promote_type(Tv, InputType), add),
        mul,
        add,
        accum;
        ndrange = size(A, 1),
    )
end
