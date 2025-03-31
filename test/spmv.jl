TEST_BACKEND = if get(ENV, "CI", "false") == "false"

    Metal.MetalBackend()  # our personal laptops
# KernelAbstractions.CPU()
else
    KernelAbstractions.CPU()
end

@testset "mul!" begin
    # Matrix-vector multiplication
    A_cpu = sprand(Float32, 10, 10, 0.5)
    B_cpu = rand(Float32, 10)
    C_cpu = A_cpu * B_cpu
    A_gpu_ell = SparseGPUMatrixELL(A_cpu, TEST_BACKEND)
    A_gpu_csr = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
    B_gpu = allocate(TEST_BACKEND, Float32, 10)
    copyto!(B_gpu, B_cpu)
    C_gpu_1 = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10)
    #semiring = Semiring(*, Monoid(+, 0.0), 0.0, 1.0)

    gpu_spmv!(C_gpu_1, A_gpu_csr, B_gpu)
    KernelAbstractions.synchronize(TEST_BACKEND)
    @allowscalar @test C_gpu_1 == C_cpu

    C_gpu_2 = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10)
    gpu_spmv!(C_gpu_2, A_gpu_ell, B_gpu)
    KernelAbstractions.synchronize(TEST_BACKEND)
    @allowscalar @test C_gpu_2 == C_cpu

    # Large matrix
    LARGE_NB = 1000
    A_cpu = sprand(Float32, LARGE_NB, LARGE_NB, 0.2)
    B_cpu = rand(Float32, LARGE_NB)
    C_cpu = A_cpu * B_cpu
    A_gpu_csr = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
    A_gpu_ell = SparseGPUMatrixELL(A_cpu, TEST_BACKEND)
    B_gpu = allocate(TEST_BACKEND, Float32, LARGE_NB)
    copyto!(B_gpu, B_cpu)
    C_gpu_1 = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB)
    C_gpu_2 = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB)

    gpu_spmv!(C_gpu_1, A_gpu_csr, B_gpu)
    gpu_spmv!(C_gpu_2, A_gpu_ell, B_gpu)
    KernelAbstractions.synchronize(TEST_BACKEND)

    # Count the number of differences
    diff = 0
    for i = 1:LARGE_NB
        if @allowscalar abs(C_gpu_1[i] - C_cpu[i]) > 1e-6 ||
                        @allowscalar abs(C_gpu_2[i] - C_cpu[i]) > 1e-6
            diff += 1
        end
    end
    @test diff <= 0
    println("Number of differences: $diff out of $LARGE_NB")

end
