
function gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixSELL{Tv,Ti},
    B::InputVec;
    mul::Function = GPUGraphs_mul,
    add::Function = GPUGraphs_add,
    accum::Function = GPUGraphs_second,
    range::Union{Nothing, UnitRange} = nothing,
    mask::Union{Nothing, AbstractVector} = nothing,
) where {
    Tv,
    Ti<:Integer,
    ResType<:Number,
    InputType<:Number,
    ResVec<:AbstractVector{ResType},
    InputVec<:AbstractVector{InputType},
}
    # call appropriate specialized function
    if mask === nothing && range === nothing
        _simple_gpu_spmv!(C, A, B, mul, add, accum)
    elseif mask !== nothing && range === nothing
        _dense_mask_gpu_spmv!(C, A, B,  mul, add, accum, mask)
    elseif mask === nothing && range !== nothing
        _range_gpu_spmv!(C, A, B,mul, add, accum, range)
    else
        _range_mask_gpu_spmv!(C, A, B, mul, add, accum, mask, range)
    end
end

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

### No mask version
function _simple_gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixSELL{Tv,Ti},
    B::InputVec,
    mul::Function,
    add::Function,
    accum::Function,
) where {
    Tv,
    Ti<:Integer,
    ResType<:Number,
    InputType<:Number,
    ResVec<:AbstractVector{ResType},
    InputVec<:AbstractVector{InputType},
}
    _validate_args(C, A, B, nothing)
    backend = get_backend(A)
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
        ndrange = size(C, 1),
    )
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

## Dense mask version
function _dense_mask_gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixSELL{Tv,Ti},
    B::InputVec,
    mul::Function,
    add::Function,
    accum::Function,
    mask::MaskVec,
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
    _validate_args(C, A, B, mask)
    backend = get_backend(A)
    zero_mask = zero(Tmask)
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
        zero_mask,
        mul,
        add,
        accum;
        ndrange = size(C, 1),
    )
end

@kernel function range_sell_spmv_kernel!(
    c,
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(a_slice_ptr),
    @Const(slice_size),
    @Const(n),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(range_start),
    mul,
    add,
    accum,
)
    #offset, slice = @index(Global, NTuple)
    #offset = offset - 1
    #row = (slice-1) * slice_size + offset + 1
    row = @index(Global, Linear) + range_start - 1
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

## Specialized UnitRange mask version
function _range_gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixSELL{Tv,Ti},
    B::InputVec,
    mul::Function,
    add::Function,
    accum::Function,
    range::UnitRange,
) where {
    Tv,
    Ti<:Integer,
    ResType<:Number,
    InputType<:Number,
    ResVec<:AbstractVector{ResType},
    InputVec<:AbstractVector{InputType},
}
    _validate_args(C, A, B, nothing)
    backend = get_backend(A)
    kernel! = range_sell_spmv_kernel!(backend)
    kernel!(
        C,
        A.colval,
        A.nzval,
        A.slice_ptr,
        A.slice_size,
        A.n,
        B,
        monoid_neutral(promote_type(Tv, InputType), add),
        first(range),
        mul,
        add,
        accum;
        ndrange = size(range, 1),
    )
end


@kernel function range_dense_masked_sell_spmv_kernel!(
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
    @Const(range_start),
    mul,
    add,
    accum,
)
    #slice, offset = @index(Global, NTuple)
    #offset = offset - 1
    #row = (slice-1) * slice_size + offset + 1
    row = @index(Global, Linear) + range_start - 1
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

## Dense mask and range version
function _range_mask_gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixSELL{Tv,Ti},
    B::InputVec,
    mul::Function,
    add::Function,
    accum::Function,
    mask::MaskVec,
    range::UnitRange,
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
    _validate_args(C, A, B, mask)
    backend = get_backend(A)
    zero_mask = zero(Tmask)
    kernel! = range_dense_masked_sell_spmv_kernel!(backend)
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
        zero_mask,
        first(range),
        mul,
        add,
        accum;
        ndrange = size(range, 1),
    )
    return
end

function _validate_args(
    C::ResVec,
    A::SparseGPUMatrixSELL{Tv,Ti},
    B::InputVec,
    mask::Union{MaskVec,Nothing},
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
    # Check dimensions
    if size(A, 2) != length(B) 
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    if size(C, 1) != size(A, 1)
        throw(DimensionMismatch("Matrix dimensions must agree"))
    end
    # Check types
    if !(promote_type(Tv, InputType) <: ResType)
        throw(ArgumentError("Result type must be able to hold the result of the multiplication"))
    end
    # Check backends
    backend = get_backend(A)
    if get_backend(B) != backend || get_backend(C) != backend
        throw(ArgumentError("All inputs must be on the same backend"))
    end
     # Check mask if provided
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
    end
end