using Metal, KernelAbstractions

backend = KernelAbstractions.CPU()

@kernel function test_kernel!(A)
    row = @index(Global, Linear)
    A[row] = row
end

function call_test_kernel()
    A = allocate(backend, Float32, 10)
    kernel! = test_kernel!(backend)
    kernel!(A; ndrange = size(A, 1))
    return A
end
