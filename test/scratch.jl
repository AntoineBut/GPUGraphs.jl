using Metal, KernelAbstractions

backend = Metal.MetalBackend()

@kernel function test_kernel!(A)
    index = KernelAbstractions.@index(Global, Linear)
    print("Hello from kernel! ", index)

end

function call_test_kernel()
    A = KernelAbstractions.zeros(backend, Int32, 10)
    #B = KernelAbstractions.ones(backend, Int32, 10)
    #C = KernelAbstractions.zeros(backend, Int32, 10)
    kernel! = test_kernel!(backend)
    kernel!(A; ndrange = size(A, 1))
    return A
end

call_test_kernel()
"""
SIZE = 10^6
FILL = 100/SIZE
BACKEND = Metal.MetalBackend()
A_csc_cpu = sprand(Float32, SIZE, SIZE, FILL)
A_csr_cpu = transpose(A_csc_cpu)
print("Converting to GPU format\n")
A_csr_gpu = SparseGPUMatrixCSR(A_csr_cpu, BACKEND)
A_ell_gpu = SparseGPUMatrixELL(A_csr_cpu, BACKEND)
b = rand(Float32, SIZE)
b_gpu = MtlVector(b)
res_gpu1 = KernelAbstractions.zeros(BACKEND, Float32, SIZE) 
res_gpu2 = KernelAbstractions.zeros(BACKEND, Float32, SIZE)
res_gpu3 = KernelAbstractions.zeros(BACKEND, Float32, SIZE)

gpu_spmv!(res_gpu1, A_ell_gpu, b_gpu)
gpu_spmv!(res_gpu1, A_ell_gpu, b_gpu)
gpu_spmv!(res_gpu1, A_ell_gpu, b_gpu)
gpu_spmv!(res_gpu1, A_ell_gpu, b_gpu)
gpu_spmv!(res_gpu1, A_ell_gpu, b_gpu)
KernelAbstractions.synchronize(BACKEND)
gpu_spmv!(res_gpu2, A_csr_gpu, b_gpu)
gpu_spmv!(res_gpu2, A_csr_gpu, b_gpu)
gpu_spmv!(res_gpu2, A_csr_gpu, b_gpu)
gpu_spmv!(res_gpu2, A_csr_gpu, b_gpu)
gpu_spmv!(res_gpu2, A_csr_gpu, b_gpu)
KernelAbstractions.synchronize(BACKEND)



SIZE = 10^5
FILL = 10/SIZE
BACKEND = Metal.MetalBackend()
A_csc_cpu = sprand(Bool, SIZE, SIZE, FILL)
A_csr_cpu = transpose(A_csc_cpu)
print("Converting to GPU format\n")
A_csr_gpu = SparseGPUMatrixCSR(A_csr_cpu, BACKEND)
A_ell_gpu = SparseGPUMatrixELL(A_csr_cpu, BACKEND)
b = rand(Bool, SIZE)
b_gpu = MtlVector(b)
res_gpu1 = KernelAbstractions.zeros(BACKEND, Bool, SIZE) 
res_gpu2 = KernelAbstractions.zeros(BACKEND, Bool, SIZE)
res_gpu3 = KernelAbstractions.zeros(BACKEND, Bool, SIZE)

gpu_spmv!(res_gpu1, A_ell_gpu, b_gpu, &, |, |)
KernelAbstractions.synchronize(BACKEND)




using GPUGraphs
using BenchmarkTools
using Metal
using KernelAbstractions
using SuiteSparseGraphBLAS
import SuiteSparseGraphBLAS: ∧, ∨

using Graphs
using GraphIO.EdgeList


MAIN_TYPE = Int32
A_T = adjacency_matrix(loadgraph("benchmark/data/italy_osm/italy_osm.mtx", EdgeListFormat()), MAIN_TYPE; dir=:in)
SIZE = size(A_T, 1)
a_ssgb = GBMatrix{MAIN_TYPE}(A_T)
A_ell_gpu = SparseGPUMatrixELL(transpose(A_T), Metal.MetalBackend())
A_csr_gpu = SparseGPUMatrixCSR(transpose(A_T), Metal.MetalBackend())

b_cpu = rand(MAIN_TYPE, SIZE)
b_gpu = MtlVector(b_cpu)

res_ssgb = GBVector(zeros(MAIN_TYPE, SIZE))
res_gpu_1 = KernelAbstractions.zeros(Metal.MetalBackend(), MAIN_TYPE, SIZE)
res_gpu_2 = KernelAbstractions.zeros(Metal.MetalBackend(), MAIN_TYPE, SIZE)


for _ in 1:10
    gpu_spmv!(res_gpu_1, A_ell_gpu, b_gpu)
end 
KernelAbstractions.synchronize(Metal.MetalBackend())

for _ in 1:10
    gpu_spmv!(res_gpu_2, A_csr_gpu, b_gpu)
end
KernelAbstractions.synchronize(Metal.MetalBackend())




gpu_spmv!(res_gpu_1, A_gpu_csr, b_gpu, &, |, |)
KernelAbstractions.synchronize(Metal.MetalBackend())
mul!(res_ssgb, a_ssgb, b, (∧, ∨); accum=∨)



	"""
