
@kernel function sell_spmv_kernel!(
    c,
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(a_slice_ptr),
    @Const(slice_size),
    @Const(n),
    @Const(b),
    @Const(monoid_neutral_element),
    mul,
    add,
    accum,
)
    #offset, slice = @index(Global, NTuple)
    #offset = offset - 1
    #row = (slice-1) * slice_size + offset + 1
    row = @index(Global, Linear)
    slice = (row-1) ÷ slice_size + 1
    offset = (row-1) % slice_size
    if row <= n
        start = a_slice_ptr[slice] + offset
        stop = a_slice_ptr[slice+1] - 1
        acc = monoid_neutral_element
        for i = start:slice_size:stop
            col = a_col_val[i]
            if col == -1
                break
            end
            acc = add(acc, mul(a_nz_val[i], b[col], row, col, col, 1), row, col, col, 1)
        end
        c[row] = accum(c[row], acc, row, 1, row, 1)
    end
end

@kernel function dense_masked_sell_spmv_kernel!(
    c,
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(a_slice_ptr),
    @Const(slice_size),
    @Const(n),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(mask),
    @Const(mask_zero),
    mul,
    add,
    accum,
)
    #slice, offset = @index(Global, NTuple)
    #offset = offset - 1
    #row = (slice-1) * slice_size + offset + 1
    row = @index(Global, Linear)
    slice = (row-1) ÷ slice_size + 1
    offset = (row-1) % slice_size
    if row <= n && mask[row] != mask_zero

        acc = monoid_neutral_element
        for i = (a_slice_ptr[slice]+offset):slice_size:(a_slice_ptr[slice+1]-1)
            col = a_col_val[i]
            if col == -1 
                break
            end
            acc = add(acc, mul(a_nz_val[i], b[col], row, col, col, 1), row, col, col, 1)
        end
        c[row] = accum(c[row], acc, row, 1, row, 1)
    end
end



function gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixSELL{Tv,Ti},
    B::InputVec;
    mul::Function = GPUGraphs_mul,
    add::Function = GPUGraphs_add,
    accum::Function = GPUGraphs_second,
    mask::Union{MaskVec,Nothing} = nothing,
) where {
    Tv,
    Ti<:Integer,
    Tmask<:Integer,
    ResType<:Number,
    InputType<:Number,
    ResVec<:AbstractVector{ResType},
    InputVec<:AbstractVector{InputType},
    MaskVec<:AbstractVector{Tmask},
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
    backend = get_backend(A)

    # Using mask 
    if mask !== nothing
        # Check mask type
        if !(typeof(mask) <: AbstractVector{Tmask})
            throw(DimensionMismatch("Mask must be a vector"))
        end
        # Check mask length
        if length(mask) != size(A, 1)
            throw(DimensionMismatch("Mask length must be equal to the number of rows in A"))
        end
        # Check mask backend
        if get_backend(mask) != backend
            throw(ArgumentError("Mask must be on the same backend as A"))
        end

        kernel! = dense_masked_sell_spmv_kernel!(backend)
        kernel!(
            C,
            A.colval,
            A.nzval,
            A.slice_ptr,
            A.slice_size,
            A.n,
            B,
            monoid_neutral(promote_type(Tv, InputType), add),
            mask,
            zero(Tmask),
            mul,
            add,
            accum;
            ndrange = size(A, 1),
        )
        return
    end

    kernel! = sell_spmv_kernel!(backend)
    kernel!(
        C,
        A.colval,
        A.nzval,
        A.slice_ptr,
        A.slice_size,
        A.n,
        B,
        monoid_neutral(promote_type(Tv, InputType), add),
        mul,
        add,
        accum;
        ndrange = size(A, 1),
    )
end
