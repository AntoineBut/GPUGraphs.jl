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

using csv

# Load csv file, for each line "a b c" write to a new file "a b" (without c)

function load_csv(original_file::String, new_file::String)
    open(original_file, "r") do file
        open(new_file, "a") do new_file
            # Skip header
            readline(file)
            for line in eachline(file)
                # Split the line by comma
                parts = split(line, " ")
                # Write the first two parts to the new file
                if parts[1] != parts[2]
                    write(new_file, join(parts[1:2], " "))
                    write(new_file, "\n")
                end

            end
        end
    end

end


SIZE = 10^6
FILL = 100 / SIZE
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
FILL = 10 / SIZE
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
graph =
    SimpleGraph(loadgraph("benchmark/data/nlpkkt160/nlpkkt160-bool.mtx", EdgeListFormat()))
A_T = adjacency_matrix(graph, MAIN_TYPE; dir = :in)
SIZE = size(A_T, 1)
a_ssgb = GBMatrix{MAIN_TYPE}(A_T)
A_ell_gpu = SparseGPUMatrixELL(transpose(A_T), Metal.MetalBackend())
A_csr_gpu = SparseGPUMatrixCSR(transpose(A_T), Metal.MetalBackend())

b_cpu = rand(MAIN_TYPE, SIZE)
b_gpu = MtlVector(b_cpu)

res_ssgb = GBVector(zeros(MAIN_TYPE, SIZE))
res_gpu_1 = KernelAbstractions.zeros(Metal.MetalBackend(), MAIN_TYPE, SIZE)
res_gpu_2 = KernelAbstractions.zeros(Metal.MetalBackend(), MAIN_TYPE, SIZE)

Metal.@capture begin
    for _ = 1:5
        gpu_spmv!(res_gpu_2, A_csr_gpu, b_gpu)
    end
    KernelAbstractions.synchronize(Metal.MetalBackend())


    for _ = 1:5
        gpu_spmv!(res_gpu_1, A_ell_gpu, b_gpu)
    end
    KernelAbstractions.synchronize(Metal.MetalBackend())
end

for _ = 1:5
    gpu_spmv!(res_gpu_2, A_csr_gpu, b_gpu)
end
KernelAbstractions.synchronize(Metal.MetalBackend())




gpu_spmv!(res_gpu_1, A_gpu_csr, b_gpu, &, |, |)
KernelAbstractions.synchronize(Metal.MetalBackend())
mul!(res_ssgb, a_ssgb, b_cpu, (∧, ∨); accum = ∨)



using Metal
using KernelAbstractions
using LinearAlgebra
BACKEND = Metal.MetalBackend()
cpu_vec = rand(Float32, 10)
cpu_vec2 = rand(Float32, 10) * 0.001 + cpu_vec

gpu_vec = KernelAbstractions.zeros(BACKEND, Float32, 10)
copyto!(gpu_vec, cpu_vec2)

println("diff on cpu: ", norm(cpu_vec2 - cpu_vec))
println("diff on gpu: ", norm(gpu_vec - cpu_vec))
