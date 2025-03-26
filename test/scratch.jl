using Metal, KernelAbstractions

backend = KernelAbstractions.CPU()

@kernel function test_kernel!(C, A, B, add, mul, accum)
    i = @index(Global, Linear)
    a = mul(A[i], B[i])
    b = add(a, B[i])
    C[i] = accum(b, C[i])

end

function call_test_kernel()
    A = KernelAbstractions.zeros(backend, Int32, 10)
    B = KernelAbstractions.ones(backend, Int32, 10)
    C = KernelAbstractions.zeros(backend, Int32, 10)
    kernel! = test_kernel!(backend)
    kernel!(C, A, B, +, *, +; ndrange = size(A, 1))
    return C
end

call_test_kernel()
