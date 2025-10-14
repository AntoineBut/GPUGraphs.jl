"""
@kernel function csr_spmm_kernel_vec!(
    C,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(B),
    @Const(monoid_neutral_element),
    @Const(terminal_value),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    row = @index(Global, Linear)
	
    #@private acc = fill(monoid_neutral_element, size(B, 2))

    for i = a_row_ptr[row]:a_row_ptr[row+1]-1
        
        col_A = a_col_val[i]
		for col_B_C in 1:size(B, 2)
        	C[row, col_B_C] = add(C[row, col_B_C], mul(a_nz_val[i], B[col_A, col_B_C], row, col_A, col_A, col_B_C), row, col_A, col_A, col_B_C)
        	if C[row, col_B_C] == terminal_value
        	    break
        	end
		end
    end

end
"""

####### CSR SpMM #######

@kernel function csr_spmm_kernel!(
    C,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(B),
    @Const(monoid_neutral_element),
    @Const(terminal_value),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    col_B_C, row = @index(Global, NTuple)
    #row, col_B_C = @index(Global, NTuple)
    acc = monoid_neutral_element
    for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)

        col_A = a_col_val[i]
        acc = add(
            acc,
            mul(a_nz_val[i], B[col_A, col_B_C], row, col_A, col_A, col_B_C),
            row,
            col_A,
            col_A,
            col_B_C,
        )
        if acc == terminal_value
            break
        end
    end
    C[row, col_B_C] = accum(C[row, col_B_C], acc, row, col_B_C, row, col_B_C)
end

@kernel function dense_masked_csr_spmm_kernel!(
    C,
    @Const(a_row_ptr),
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(B),
    @Const(monoid_neutral_element),
    @Const(terminal_value),
    @Const(mask),
    @Const(mask_zero),
    mul,
    add,
    accum,
)
    # Computes A*B and stores the result in C
    col_B_C, row = @index(Global, NTuple)

    if mask[row] != mask_zero
        acc = monoid_neutral_element
        for i = a_row_ptr[row]:(a_row_ptr[row+1]-1)

            col_A = a_col_val[i]
            acc = add(
                acc,
                mul(a_nz_val[i], B[col_A, col_B_C], row, col_A, col_A, col_B_C),
                row,
                col_A,
                col_A,
                col_B_C,
            )
            if acc == terminal_value
                break
            end
        end
        C[row, col_B_C] = accum(C[row, col_B_C], acc, row, col_B_C, row, col_B_C)
    end
end

function gpu_spmm!(
    C::ResMat,
    A::SparseGPUMatrixCSR{Tv,Ti},
    B::InputMat;
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
    ResMat<:AbstractMatrix{ResType},
    InputMat<:AbstractMatrix{InputType},
    MaskVec<:AbstractVector{Tmask},
}
    # C is a dense matrix
    @assert size(A, 2) == size(B, 1)
    @assert size(C, 1) == size(A, 1)
    @assert size(C, 2) == size(B, 2)

    backend = get_backend(A)

    if mask === nothing

        kernel! = csr_spmm_kernel!(backend)
        #kernel! = csr_spmm_kernel_vec!(backend)
        kernel!(
            C,
            A.rowptr,
            A.colval,
            A.nzval,
            B,
            monoid_neutral(Tv, add),
            monoid_absorb(Tv, add),
            mul,
            add,
            accum,
            #ndrange = (size(A, 1),) # linear
            ndrange = (size(B, 2), size(A, 1)),
            #ndrange = (size(A, 1), size(B, 2))
        )
        return
    end

    @assert length(mask) == size(A, 1)
    mask_zero = zero(eltype(mask))

    kernel! = dense_masked_csr_spmm_kernel!(backend)
    kernel!(
        C,
        A.rowptr,
        A.colval,
        A.nzval,
        B,
        monoid_neutral(Tv, add),
        monoid_absorb(Tv, add),
        mask,
        mask_zero,
        mul,
        add,
        accum,
        ndrange = (size(B, 2), size(A, 1)),
    )

    return
end

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
            if col_A == -1
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
            if acc == terminal_value
                break
            end
        end
        C[row, col_B_C] = accum(C[row, col_B_C], acc, row, col_B_C, row, col_B_C)
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
        monoid_neutral(Tv, add),
        monoid_absorb(Tv, add),
        mul,
        add,
        accum,
        ndrange = (size(B, 2), A.slice_size, A.nslices),
    )

    return
end
