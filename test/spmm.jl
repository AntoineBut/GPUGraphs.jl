TEST_BACKEND = if get(ENV, "CI", "false") == "false"

    #Metal.MetalBackend()  # our personal laptops
    CUDA.CUDABackend()  # on the cluster
    #KernelAbstractions.CPU()
else
    KernelAbstractions.CPU()
end

@testset "mul!" begin
	# Matrix-matrix multiplication
	A_cpu = sprand(Float32, 10, 10, 0.5)
	B_cpu = rand(Float32, 10, 5)
	C_cpu = A_cpu * B_cpu
	res = zeros(Float32, 10, 5)
	A_gpu_sell = SparseGPUMatrixSELL(A_cpu, TEST_BACKEND)
	A_gpu_csr = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
	#A_gpu_csc = SparseGPUMatrixCSC(A_cpu, TEST_BACKEND)
	
	B_gpu = allocate(TEST_BACKEND, Float32, 10, 5)

	copyto!(B_gpu, B_cpu)
	C_gpu_1 = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10, 5)

	# CSR
	gpu_spmm!(C_gpu_1, A_gpu_csr, B_gpu)
	KernelAbstractions.synchronize(TEST_BACKEND)
	copyto!(res, C_gpu_1)
	@test isapprox(res, C_cpu)

	# SELL
	C_gpu_2 = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10, 5)
	gpu_spmm!(C_gpu_2, A_gpu_sell, B_gpu)
	KernelAbstractions.synchronize(TEST_BACKEND)
	copyto!(res, C_gpu_2)
	@test isapprox(res, C_cpu)

	# CSC: not yet implemented
	"""
	C_gpu_3 = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10, 5)
	gpu_spmm!(C_gpu_3, A_gpu_csc, B_gpu)
	KernelAbstractions.synchronize(TEST_BACKEND)
	copyto!(res, C_gpu_3)
	@test isapprox(res, C_cpu)
	"""
	# Large matrix
	LARGE_NB = 1000
	A_cpu = sprand(Float32, LARGE_NB, LARGE_NB, 0.2)
	B_cpu = rand(Float32, LARGE_NB, 20)
	C_cpu = A_cpu * B_cpu
	A_gpu_csr = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
	A_gpu_sell = SparseGPUMatrixSELL(A_cpu, TEST_BACKEND)
	#A_gpu_csc = SparseGPUMatrixCSC(A_cpu, TEST_BACKEND)
	
	B_gpu = allocate(TEST_BACKEND, Float32, LARGE_NB, 20)
	copyto!(B_gpu, B_cpu)
	C_gpu_1 = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB, 20)
	C_gpu_2 = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB, 20)
	#C_gpu_3 = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB, 20)
	gpu_spmm!(C_gpu_1, A_gpu_csr, B_gpu)
	#gpu_spmm!(C_gpu_2, A_gpu_sell, B_gpu)
	#gpu_spmm!(C_gpu_3, A_gpu_csc, B_gpu)
	KernelAbstractions.synchronize(TEST_BACKEND)

	res = zeros(Float32, LARGE_NB, 20)
	copyto!(res, C_gpu_1)
	@test isapprox(res, C_cpu)
	#copyto!(res, C_gpu_2)
	@test isapprox(res, C_cpu)
	#copyto!(res, C_gpu_3)
	#@test isapprox(res, C_cpu)
	
end