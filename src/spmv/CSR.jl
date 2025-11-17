
function gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixCSR{Tv,Ti},
    B::InputVec;
    mul::Function = GPUGraphs_mul,
    add::Function = GPUGraphs_add,
    accum::Function = GPUGraphs_second,
    range::Union{Nothing,UnitRange} = nothing,
    mask::Union{Nothing,AbstractVector} = nothing,
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
        _dense_mask_gpu_spmv!(C, A, B, mul, add, accum, mask)
    elseif mask === nothing && range !== nothing
        _range_gpu_spmv!(C, A, B, mul, add, accum, range)
    else
        _range_mask_gpu_spmv!(C, A, B, mul, add, accum, mask, range)
    end
end

@kernel function csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(monoid_absorb),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    row = @index(Global, Linear)
    acc = monoid_neutral_element
    for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)
        col = a_col_val[i]
        acc = add(acc, mul(a_nz_val[i], b[col], row, col, col, 1), row, col, col, 1)
        if acc == monoid_absorb
            break
        end
    end
    c[row] = accum(c[row], acc, row, 1, row, 1)
end

function _simple_gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixCSR{Tv,Ti},
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
    csr_spmv_kernel!(backend)(
        C,
        A.rowptr,
        A.colval,
        A.nzval,
        B,
        monoid_neutral(promote_type(Tv, InputType), add),
        monoid_absorb(promote_type(Tv, InputType), add),
        mul,
        add,
        accum;
        ndrange = size(C, 1),
    )
    return
end


@kernel function sparse_masked_csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(monoid_absorb),
    @Const(mask),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    entry_nb = @index(Global, Linear)
    row = mask[entry_nb]
    acc = monoid_neutral_element
    for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)
        col = a_col_val[i]
        acc = add(acc, mul(a_nz_val[i], b[col], row, col, col, 1), row, col, col, 1)
        if acc == monoid_absorb
            break
        end
    end
    c[row] = accum(c[row], acc, row, 1, row, 1)
end

@kernel function dense_masked_csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(monoid_absorb),
    @Const(mask),
    @Const(mask_zero),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    row = @index(Global, Linear)
    if mask[row] != mask_zero
        acc = monoid_neutral_element
        for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)
            col = a_col_val[i]
            acc = add(acc, mul(a_nz_val[i], b[col], row, col, col, 1), row, col, col, 1)
            if acc == monoid_absorb
                break
            end
        end
        c[row] = accum(c[row], acc, row, 1, row, 1)
    end
end


@kernel function any_dense_masked_csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(mask),
    @Const(mask_zero),
    mul,
    accum,
)
    # Computes A*B and stores the result in C
    row = @index(Global, Linear)
    if mask[row] != mask_zero
        for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)
            col = a_col_val[i]
            b_val = b[col]
            if b_val != zero(b_val)
                c[row] =
                    accum(c[row], mul(a_nz_val[i], b_val, row, col, col, 1), row, 1, row, 1)
                break
            end
        end
    end
end

function _dense_mask_gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixCSR{Tv,Ti},
    B::InputVec,
    mul::Function,
    add::Function,
    accum::Function,
    mask::AbstractVector,
) where {
    Tv,
    Ti<:Integer,
    ResType<:Number,
    InputType<:Number,
    ResVec<:AbstractVector{ResType},
    InputVec<:AbstractVector{InputType},
}
    _validate_args(C, A, B, mask)
    backend = get_backend(A)
    if add == GPUGraphs_any #TODO: make a specialized any version for other kernels
        any_dense_masked_csr_spmv_kernel!(backend)(
            C,
            A.rowptr,
            A.colval,
            A.nzval,
            B,
            mask,
            zero(eltype(mask)),
            mul,
            accum;
            ndrange = size(C, 1),
        )
        return
    end
    dense_masked_csr_spmv_kernel!(backend)(
        C,
        A.rowptr,
        A.colval,
        A.nzval,
        B,
        monoid_neutral(promote_type(Tv, InputType), add),
        monoid_absorb(promote_type(Tv, InputType), add),
        mask,
        zero(eltype(mask)),
        mul,
        add,
        accum;
        ndrange = size(C, 1),
    )
    return
end

@kernel function range_csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(monoid_absorb),
    @Const(range_start),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    row = @index(Global, Linear) + range_start - 1
    acc = monoid_neutral_element
    for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)
        col = a_col_val[i]
        acc = add(acc, mul(a_nz_val[i], b[col], row, col, col, 1), row, col, col, 1)
        if acc == monoid_absorb
            break
        end
    end
    c[row] = accum(c[row], acc, row, 1, row, 1)
end

function _range_gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixCSR{Tv,Ti},
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
    range_csr_spmv_kernel!(backend)(
        C,
        A.rowptr,
        A.colval,
        A.nzval,
        B,
        monoid_neutral(promote_type(Tv, InputType), add),
        monoid_absorb(promote_type(Tv, InputType), add),
        range.start,
        mul,
        add,
        accum;
        ndrange = size(range, 1),
    )
    return
end

@kernel function range_masked_csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(monoid_absorb),
    @Const(mask),
    @Const(mask_zero),
    @Const(range_start),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    row = @index(Global, Linear) + range_start - 1
    if mask[row] != mask_zero
        acc = monoid_neutral_element
        for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)
            col = a_col_val[i]
            acc = add(acc, mul(a_nz_val[i], b[col], row, col, col, 1), row, col, col, 1)
            if acc == monoid_absorb
                break
            end
        end
        c[row] = accum(c[row], acc, row, 1, row, 1)
    end
end



function _range_mask_gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixCSR{Tv,Ti},
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
    range_masked_csr_spmv_kernel!(backend)(
        C,
        A.rowptr,
        A.colval,
        A.nzval,
        B,
        monoid_neutral(promote_type(Tv, InputType), add),
        monoid_absorb(promote_type(Tv, InputType), add),
        mask,
        zero(eltype(mask)),
        range.start,
        mul,
        add,
        accum;
        ndrange = size(range, 1),
    )
    return
end


function _validate_args(
    C::ResVec,
    A::SparseGPUMatrixCSR{Tv,Ti},
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
        throw(
            ArgumentError(
                "Result type must be able to hold the result of the multiplication",
            ),
        )
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
