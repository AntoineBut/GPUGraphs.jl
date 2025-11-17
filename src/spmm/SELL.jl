######## SELL SpMM #######
@kernel function sell_spmm_kernel!(
    C,
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(a_slice_ptr),
    @Const(a_slice_size),
    @Const(B),
    @Const(n),
    @Const(monoid_neutral_element),
    @Const(terminal_value),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    col_B_C, offset, slice = @index(Global, NTuple)
    offset = offset - 1
    row = (slice-1) * a_slice_size + offset + 1
    acc = monoid_neutral_element
    if row <= n
        for i = (a_slice_ptr[slice]+offset):a_slice_size:(a_slice_ptr[slice+1]-1)

            col_A = a_col_val[i]
            if col_A == -1 || acc == terminal_value
                break
            end
            acc = add(
                acc,
                mul(a_nz_val[i], B[col_A, col_B_C], row, col_A, col_A, col_B_C),
                row,
                col_A,
                col_A,
                col_B_C,
            )
        end
        C[row, col_B_C] = accum(C[row, col_B_C], acc, row, col_B_C, row, col_B_C)
    end
end

@kernel function dense_masked_sell_spmm_kernel!(
    C,
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(a_slice_ptr),
    @Const(a_slice_size),
    @Const(B),
    @Const(n),
    @Const(monoid_neutral_element),
    @Const(terminal_value),
    @Const(mask),
    @Const(mask_zero),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    col_B_C, offset, slice = @index(Global, NTuple)
    offset = offset - 1
    row = (slice-1) * a_slice_size + offset + 1
    if mask[row] != mask_zero
        acc = monoid_neutral_element
        if row <= n
            for i = (a_slice_ptr[slice]+offset):a_slice_size:(a_slice_ptr[slice+1]-1)
                col_A = a_col_val[i]
                if col_A == -1 || acc == terminal_value
                    break
                end
                acc = add(
                    acc,
                    mul(a_nz_val[i], B[col_A, col_B_C], row, col_A, col_A, col_B_C),
                    row,
                    col_A,
                    col_A,
                    col_B_C,
                )

            end
            C[row, col_B_C] = accum(C[row, col_B_C], acc, row, col_B_C, row, col_B_C)
        end
    end
end

function gpu_spmm!(
    C::ResMat,
    A::SparseGPUMatrixSELL{Tv,Ti},
    B::InputMat;
    mul::Function = GPUGraphs_mul,
    add::Function = GPUGraphs_add,
    accum::Function = GPUGraphs_second,
) where {
    Tv,
    Ti<:Integer,
    ResType<:Number,
    InputType<:Number,
    ResMat<:AbstractMatrix{ResType},
    InputMat<:AbstractMatrix{InputType},
}
    # C is a dense matrix
    @assert size(A, 2) == size(B, 1)
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)

    backend = get_backend(A)

    kernel! = sell_spmm_kernel!(backend)
    kernel!(
        C,
        A.colval,
        A.nzval,
        A.slice_ptr,
        A.slice_size,
        B,
        size(A, 1),
        monoid_neutral(promote_type(Tv, InputType), add),
        monoid_absorb(promote_type(Tv, InputType), add),
        mul,
        add,
        accum,
        ndrange = (size(B, 2), A.slice_size, A.nslices),
    )

    return
end
