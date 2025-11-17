@kernel function csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(terminal_value),
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
        if acc == terminal_value
            break
        end
    end
    c[row] = accum(c[row], acc, row, 1, row, 1)
end

@kernel function sparse_masked_csr_spmv_kernel!(
    c,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(terminal_value),
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
        if acc == terminal_value
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
    @Const(terminal_value),
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
            if acc == terminal_value
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

function gpu_spmv!(
    C::ResVec,
    A::SparseGPUMatrixCSR{Tv,Ti},
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
    backend = get_backend(C)

    #println("monoid absorb $add for $Tv: ", monoid_absorb(Tv, add))
    # No mask
    if mask === nothing
        kernel! = csr_spmv_kernel!(backend)
        kernel!(
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
            ndrange = size(A, 1),
        )
        return
    end
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

    # SparseVector mask 
    if typeof(mask) <: AbstractSparseGPUVector{Tmask,Ti}
        kernel! = sparse_masked_csr_spmv_kernel!(backend)
        kernel!(
            C,
            A.rowptr,
            A.colval,
            A.nzval,
            B,
            monoid_neutral(promote_type(Tv, InputType), add),
            monoid_absorb(promote_type(Tv, InputType), add),
            mask.nzind,
            mul,
            add,
            accum;
            ndrange = nnz(mask),
        )
        return
    end

    # DenseVector mask
    if typeof(mask) <: AbstractVector{Tmask}
        if add == GPUGraphs_any
            #println("Using any_dense_masked_csr_spmv_kernel!")
            kernel! = any_dense_masked_csr_spmv_kernel!(backend)
            kernel!(
                C,
                A.rowptr,
                A.colval,
                A.nzval,
                B,
                mask,
                zero(Tmask),
                mul,
                accum;
                ndrange = size(A, 1),
            )
            return
        end
        kernel! = dense_masked_csr_spmv_kernel!(backend)
        kernel!(
            C,
            A.rowptr,
            A.colval,
            A.nzval,
            B,
            monoid_neutral(promote_type(Tv, InputType), add),
            monoid_absorb(promote_type(Tv, InputType), add),
            mask,
            zero(Tmask),
            mul,
            add,
            accum;
            ndrange = size(A, 1),
        )
        return
    end

end