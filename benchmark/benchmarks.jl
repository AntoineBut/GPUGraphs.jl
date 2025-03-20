using GPUGraphs
using BenchmarkTools
using SparseArrays
import SuiteSparseGraphBLAS: GBMatrix, GBVector, gbrand
using Metal
using KernelAbstractions
import LinearAlgebra: mul!


SUITE = BenchmarkGroup()
SUITE["rand"] = @benchmarkable rand(10)

# Write your benchmarks here.
BACKEND = Metal.MetalBackend()  # our personal laptops
# Get number of CPU threads
n_cpu_threads = Sys.CPU_THREADS

SIZE = 1024*16
FILL = 0.2
print("Generating random sparse matrix of size $SIZE x $SIZE with fill $FILL\n")
A_csc_cpu = sprand(Float32, SIZE, SIZE, FILL)
A_csr_cpu = transpose(A_csc_cpu)
print("Building GB sparse matrix\n")
A_ssGB = gbrand(Float32, SIZE, SIZE, FILL)



print("Converting to GPU format\n")
A_csr_gpu = SparseGPUMatrixCSR(A_csr_cpu, BACKEND)
print("Done. \n Generating random vector of size $SIZE\n")

b = rand(Float32, SIZE)
b_ssGB = b
b_gpu = MtlVector(b)

res = zeros(Float32, SIZE)
res_ssGB = GBVector(res)
res_gpu = allocate(BACKEND, Float32, SIZE)

semiring = Semiring((x, y) -> x * y, Monoid(+, 0.0), 0.0, 1.0)

print("Benchmarking mul! on CPU and GPU\n")
SUITE["mul!"]["CPU"]["SparseArrays-CSR"] = @benchmarkable begin
	mul!(res, A_csr_cpu, b)
end

SUITE["mul!"]["CPU"]["SparseArrays-CSC"] = @benchmarkable begin
	mul!(res, A_csc_cpu, b)
end
SUITE["mul!"]["CPU"]["SuiteSparseGraphBLAS"] = @benchmarkable begin
	mul!(res_ssGB, A_ssGB, b_ssGB)
end
SUITE["mul!"]["GPU"] = @benchmarkable begin
	GPU_spmul!(res_gpu, A_csr_gpu, b_gpu, semiring)
	KernelAbstractions.synchronize(BACKEND)
end



