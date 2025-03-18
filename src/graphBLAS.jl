# This files contains implementations of GraphBLAS operations for sparse matrices and vectors.


# Priority : efficient elementwise operations using mapreduce, Matrix-Vector products, Matrix-Matrix products with GraphBLAS semirings


@kernel function row_mul_kernel!(
	c,
    a_row_ptr,
    a_col_val,
    a_nz_val,
    b,
    semiring::Semiring,
)
    # Computes A*B and stores the result in C using the semiring semiring.
	@private row = @index(Global)
    @print row
    for i in a_row_ptr[row]:a_row_ptr[row+1]-1
        c[row] += b[a_col_val[i]] * a_nz_val[i]
    end

end

function mul!(
    C::AV,
    A::SparseGPUMatrixCSR,
    B::AV,
    semiring::Semiring,
) where {AV<:AbstractGPUVector}
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
    kernel! = row_mul_kernel!(backend, )
	print(typeof(A.nzval))
    kernel!(C, A.rowptr, A.colval, A.nzval, B, semiring, ndrange=size(C))
end
