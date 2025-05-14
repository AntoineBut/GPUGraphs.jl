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

using SuiteSparseGraphBLAS
import SuiteSparseGraphBLAS: ∧, ∨
using ParallelGraphs


using GPUGraphs
using BenchmarkTools
using Metal
using KernelAbstractions


using Graphs
using GraphIO.EdgeList


MAIN_TYPE = Bool
graph = SimpleGraph(loadgraph("benchmark/data/com-Orkut/com-Orkut.mtx", EdgeListFormat()))
#graph = dorogovtsev_mendes(10)
A = convert(
    SparseMatrixCSC{MAIN_TYPE,Int32},
    adjacency_matrix(graph, MAIN_TYPE; dir = :out),
)
SIZE = size(A, 1)

A_T_gpu = SparseGPUMatrixCSR(transpose(A), Metal.MetalBackend())
A_T = GBMatrix{Bool}((adjacency_matrix(graph, Bool; dir=:in))) 

p = GBVector{Int}(SIZE; fill=zero(Int))

Metal.@capture begin
#@benchmark begin
    GPUGraphs.bfs_distances(A_T_gpu, Int32(1))
    KernelAbstractions.synchronize(Metal.MetalBackend())
    GPUGraphs.bfs_parents(A_T_gpu, Int32(1))
    KernelAbstractions.synchronize(Metal.MetalBackend())
end
#end

@benchmark begin
    GPUGraphs.bfs_distances(A_T_gpu, Int32(1))
    KernelAbstractions.synchronize(Metal.MetalBackend())
end 

@benchmark begin
    GPUGraphs.bfs_parents(A_T_gpu, Int32(1))
    KernelAbstractions.synchronize(Metal.MetalBackend())
end

@benchmark begin
    gdistances(graph, 1)
end
@benchmark begin
    Graphs.bfs_parents(graph, 1)
end

@benchmark begin
    ParallelGraphs.bfs_BLAS!(A_T, 1, p)
end eval = 1 setup = (p = GBVector{Int}(SIZE; fill=zero(Int)))
