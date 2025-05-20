# This files contains implementations of GraphBLAS operations for sparse matrices and vectors.

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
    for i = a_row_ptr[row]:a_row_ptr[row+1]-1
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
    for i = a_row_ptr[row]:a_row_ptr[row+1]-1
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
        for i = a_row_ptr[row]:a_row_ptr[row+1]-1
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
        for i = a_row_ptr[row]:a_row_ptr[row+1]-1
            col = a_col_val[i]
            b_val = b[col]
            if b_val != zero(b_val)
                c[row] = accum(c[row], mul(a_nz_val[i], b_val, row, col, col, 1), row, 1, row, 1)
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
            monoid_neutral(Tv, add),
            monoid_absorb(Tv, add),
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
            monoid_neutral(Tv, add),
            monoid_absorb(Tv, add),
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
            monoid_neutral(Tv, add),
            monoid_absorb(Tv, add),
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
    for i = a_col_ptr[col]:a_col_ptr[col+1]-1
        row = a_row_val[i]
        acc = mul(b[col], a_nz_val[i], row, col, col, 1)
        Atomix.@atomic c[row] += acc
    end
end

function gpu_spmv!(
    C::AV,
    A::SparseGPUMatrixCSC{Tv,Ti},
    B::AV;
    mul::Function = GPUGraphs_mul,
    add::Function = GPUGraphs_add,
    accum::Function = GPUGraphs_second,
) where {Tv,Ti,AV<:AbstractVector{Tv}}
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
        monoid_neutral(Tv, add),
        mul,
        add,
        accum;
        ndrange = size(A, 1),
    )
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
    offset, slice = @index(Global, NTuple)
    offset = offset - 1
    row = (slice-1) * slice_size + offset + 1

    #row = @index(Global, Linear)
    #slice = (row-1) ÷ slice_size + 1
    #offset = (row-1) % slice_size
    if row <= n
        start = a_slice_ptr[slice] + offset
        stop = a_slice_ptr[slice + 1] - 1
        acc = monoid_neutral_element
        for i = start:slice_size:stop
            #if i > length(a_nz_val)
            #    break
            #end
            col = a_col_val[i]
            if col == 0 # This is a padding value. The remaining values are all 0
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
    @Const(a_nnz_per_row),
    @Const(n),
    @Const(b),
    @Const(monoid_neutral_element),
    @Const(mask),
    @Const(mask_zero),
    mul,
    add,
    accum,
)
    offset, slice = @index(Global, NTuple)
    offset = offset - 1
    row = (slice-1) * slice_size + offset + 1

    if row <= n && mask[row] != mask_zero
        start = a_slice_ptr[slice] + offset
        stop = a_slice_ptr[slice + 1] - 1
        acc = monoid_neutral_element
        for i = start:slice_size:stop
            #if i > length(a_nz_val)
            #    break
            #end
            col = a_col_val[i]
            if col == 0 # This is a padding value. The remaining values are all 0
                break
            end
            acc = add(acc, mul(a_nz_val[i], b[col], row, col, col, 1), row, col, col, 1)
        end
        c[row] = accum(c[row], acc, row, 1, row, 1)
    end
end

@kernel function any_dense_masked_sell_spmv_kernel!(
    c,
    @Const(a_col_val),
    @Const(a_nz_val),
    @Const(a_slice_ptr),
    @Const(slice_size),
    @Const(n),
    @Const(b),
    @Const(mask),
    @Const(mask_zero),
    mul,
    accum,
)
    offset, slice = @index(Global, NTuple)
    offset = offset - 1
    row = (slice-1) * slice_size + offset + 1

    if row <= n && mask[row] != mask_zero
        start = a_slice_ptr[slice] + offset
        stop = a_slice_ptr[slice + 1] - 1
        for i = start:slice_size:stop
            #if i > length(a_nz_val)
            #    break
            #end
            col = a_col_val[i]
            if col == 0 # This is a padding value. The remaining values are all 0
                break
            end
            b_val = b[col]
            if b_val != zero(b_val)
                c[row] = accum(c[row], mul(a_nz_val[i], b_val, row, col, col, 1), row, 1, row, 1)
                break
            end
        end
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

    # No mask
    if mask === nothing
        kernel! = sell_spmv_kernel!(backend)
        kernel!(
            C,
            A.colval,
            A.nzval,
            A.slice_ptr,
            A.slice_size,
            A.n,
            B,
            monoid_neutral(Tv, add),
            mul,
            add,
            accum;
            ndrange = (A.slice_size, A.nslices),
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

    # Any operator 
    if add == GPUGraphs_any
        kernel! = any_dense_masked_sell_spmv_kernel!(backend)
        kernel!(
            C,
            A.colval,
            A.nzval,
            A.slice_ptr,
            A.slice_size,
            A.n,
            B,
            mask,
            zero(Tmask),
            mul,
            accum;
            ndrange = (A.slice_size, A.nslices),
        )
        return
    end

    kernel! = dense_masked_sell_spmv_kernel!(backend)
    kernel!(
        C,
        A.colval,
        A.nzval,
        A.slice_ptr,
        A.slice_size,
        B,
        monoid_neutral(Tv, add),
        mask,
        zero(Tmask),
        mul,
        add,
        accum;
        ndrange = (A.slice_size, A.nslices),
    )
    return
    

    
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
