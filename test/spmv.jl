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
    println("res: ", C_cpu)
    res = zeros(Float32, 10)
    A_gpu_ell = SparseGPUMatrixELL(A_cpu, TEST_BACKEND)
    A_gpu_csr = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
    A_gpu_csc = SparseGPUMatrixCSC(A_cpu, TEST_BACKEND)
    B_gpu = allocate(TEST_BACKEND, Float32, 10)

    copyto!(B_gpu, B_cpu)
    C_gpu_1 = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10)

    gpu_spmv!(C_gpu_1, A_gpu_csr, B_gpu)
    KernelAbstractions.synchronize(TEST_BACKEND)
    @allowscalar @test C_gpu_1 == C_cpu

    C_gpu_2 = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10)
    gpu_spmv!(C_gpu_2, A_gpu_ell, B_gpu)
    KernelAbstractions.synchronize(TEST_BACKEND)
    @allowscalar @test C_gpu_2 == C_cpu

    C_gpu_3 = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10)
    gpu_spmv!(C_gpu_3, A_gpu_csc, B_gpu)
    KernelAbstractions.synchronize(TEST_BACKEND)
    copyto!(res, C_gpu_3)
    println("C_gpu_3: ", res)
    @test isapprox(res, C_cpu)

    # Large matrix
    LARGE_NB = 1000
    A_cpu = sprand(Float32, LARGE_NB, LARGE_NB, 0.2)
    B_cpu = rand(Float32, LARGE_NB)
    C_cpu = A_cpu * B_cpu
    A_gpu_csr = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
    A_gpu_ell = SparseGPUMatrixELL(A_cpu, TEST_BACKEND)
    A_gpu_csc = SparseGPUMatrixCSC(A_cpu, TEST_BACKEND)
    B_gpu = allocate(TEST_BACKEND, Float32, LARGE_NB)
    copyto!(B_gpu, B_cpu)
    C_gpu_1 = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB)
    C_gpu_2 = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB)
    C_gpu_3 = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB)

    gpu_spmv!(C_gpu_1, A_gpu_csr, B_gpu)
    gpu_spmv!(C_gpu_2, A_gpu_ell, B_gpu)
    gpu_spmv!(C_gpu_3, A_gpu_csc, B_gpu)
    KernelAbstractions.synchronize(TEST_BACKEND)

    # Count the number of differences
    diff_1 = 0
    diff_2 = 0
    diff_3 = 0.0

    for i = 1:LARGE_NB
        @allowscalar diff_1 += isapprox(C_gpu_1[i], C_cpu[i])
        @allowscalar  diff_2 += isapprox(C_gpu_2[i], C_cpu[i])
        @allowscalar diff_3 += C_gpu_3[i] - C_cpu[i]
    end
    @test diff_1 <= 0
    @test diff_2 <= 0
    @test diff_3 <= 1e-5
    println("Number of differences: $diff_1, $diff_2 out of $LARGE_NB")
    println("Approximation error on CSC: $diff_3 out of $LARGE_NB")
    println("C_gpu_3: \n", C_gpu_3[1:10])
    println("C_cpu: \n", C_cpu[1:10])

end
