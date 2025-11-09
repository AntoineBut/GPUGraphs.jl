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

gpu_spmv!(res_gpu1,A_ell_gpu,b_gpu,&,|,|)
KernelAbstractions.synchronize(BACKEND)

using SuiteSparseGraphBLAS
import SuiteSparseGraphBLAS: ∧, ∨
using ParallelGraphs

using Revise

using SparseArrays
using GPUGraphs
using BenchmarkTools
using CUDA
using KernelAbstractions
using LinearAlgebra

using Graphs
using GraphIO.EdgeList
using SuiteSparseMatrixCollection
using HarwellRutherfordBoeing


MAIN_TYPE = Bool
#MAIN_TYPE = Float32
RES_TYPE = ifelse(MAIN_TYPE == Bool, Int32, MAIN_TYPE)
BACKEND = CUDA.CUDABackend()

use_dataset = true
graph = dorogovtsev_mendes(150)
if use_dataset
    ssmc = ssmc_db();
    orkut_path = fetch_ssmc(ssmc_matrices(ssmc, "SNAP", "Orkut"), format = "RB")[1]
    nlpkkt_path = fetch_ssmc(ssmc_matrices(ssmc, "Schenk", "nlpkkt160"), format = "RB")[1]

    #loaded_matrix = RutherfordBoeingData(joinpath(orkut_path, "com-Orkut.rb"));
    loaded_matrix = RutherfordBoeingData(joinpath(nlpkkt_path, "nlpkkt160.rb"));
    graph = SimpleDiGraph(loaded_matrix.data)
else
    graph = dorogovtsev_mendes(150)
end
A_cpu = transpose(convert(
        SparseMatrixCSC{MAIN_TYPE,Int32},
        adjacency_matrix(graph, MAIN_TYPE; dir = :both),
    ))



SIZE = size(A_cpu, 2)
A_T = SparseGPUMatrixSELL(A_cpu, BACKEND)
A_T2 = SparseGPUMatrixCSR(A_cpu, BACKEND)

SIZE_2 = 8

B_cpu = rand(MAIN_TYPE, SIZE, SIZE_2);
b_cpu = B_cpu[:, 1];
C_cpu = zeros(RES_TYPE, SIZE, SIZE_2);


B = KernelAbstractions.zeros(BACKEND, MAIN_TYPE, SIZE, SIZE_2);
copyto!(B, B_cpu);
b = KernelAbstractions.zeros(BACKEND, MAIN_TYPE, SIZE);
copyto!(b, b_cpu);
C = KernelAbstractions.zeros(BACKEND, RES_TYPE, SIZE, SIZE_2);
c = KernelAbstractions.zeros(BACKEND, RES_TYPE, SIZE);

mask = rand(Bool, SIZE)
mask_dense = KernelAbstractions.zeros(BACKEND, Bool, SIZE)
copyto!(mask_dense, mask)

@benchmark begin
    mul!(C_cpu, A_cpu, B_cpu)
end

@benchmark begin
    gpu_spmm!(C, A_T2, B)

    CUDA.synchronize()
end

C_res = zeros(MAIN_TYPE, SIZE, SIZE_2);
copyto!(C_res, C);
isapprox(C_cpu, C_res)

@benchmark begin
    for _ = 1:SIZE_2
        gpu_spmv!(c, A_T, b, mask = mask_dense)
    end
    CUDA.synchronize()
end
@benchmark begin
    for _ = 1:SIZE_2
        gpu_spmv!(c, A_T2, b, mask = mask_dense)
    end
    CUDA.synchronize()
end

c_res = zeros(RES_TYPE, SIZE);
copyto!(c_res, c);
c_cpu = A_cpu * b_cpu .* mask;
isapprox(c_cpu, c_res)


@benchmark begin
    for i = 1:5
        res1 = GPUGraphs.bfs_distances(A_T_gpu2, Int32(1))
    end
    KernelAbstractions.synchronize(BACKEND)
end
mat_res = zeros(Int32, SIZE, SIZE_2)
vec_res = zeros(Int32, SIZE, SIZE_2)

### quick test
mat_res = GPUGraphs.shortest_path(A_T, convert(Vector{Int32}, range(1, SIZE_2)));
for i = 1:SIZE_2
    temp = GPUGraphs.shortest_path(A_T, Int32(i));
    temp_cpu = zeros(Int32, SIZE)
    copyto!(temp_cpu, temp)
    vec_res[:, i] = temp_cpu
end
KernelAbstractions.synchronize(BACKEND)

mat_res_cpu = zeros(Int32, SIZE, SIZE_2)
copyto!(mat_res_cpu, mat_res)
isapprox(mat_res_cpu, vec_res)



@benchmark begin
    mat_res = GPUGraphs.shortest_path(A_T, convert(Vector{Int32}, range(1, SIZE_2)));
    KernelAbstractions.synchronize(BACKEND)
end

@benchmark begin
    for i = 1:SIZE_2
        temp = GPUGraphs.shortest_path(A_T, Int32(i));
        KernelAbstractions.synchronize(BACKEND)
    end
end


A_T = GBMatrix{Bool}((adjacency_matrix(graph, Bool; dir = :in)))

p = GBVector{Int}(SIZE; fill = zero(Int))

Metal.@capture begin
    #@benchmark begin
    GPUGraphs.bfs_distances(A_T_gpu, Int32(1))
    KernelAbstractions.synchronize(BACKEND)
    GPUGraphs.bfs_parents(A_T_gpu, Int32(1))
    KernelAbstractions.synchronize(BACKEND)
end
#end

@benchmark begin
    GPUGraphs.bfs_distances(A_T_gpu, Int32(1))
    KernelAbstractions.synchronize(BACKEND)
end

@benchmark begin
    GPUGraphs.bfs_parents(A_T_gpu, Int32(1))
    KernelAbstractions.synchronize(BACKEND)
end

@benchmark begin
    gdistances(graph, 1)
end
@benchmark begin
    Graphs.bfs_parents(graph, 1)
end

@benchmark begin
    ParallelGraphs.bfs_BLAS!(A_T, 1, p)
end eval = 1 setup = (p = GBVector{Int}(SIZE; fill = zero(Int)))


using CUDA
# Stress GPU test_kernel


SIZE = 32000
A = CUDA.rand(Float32, SIZE, SIZE)
B = CUDA.rand(Float32, SIZE, SIZE)
res = CUDA.zeros(Float32, SIZE, SIZE)

@time for i = 1:100
    @. res = exp(A) * sin(B) + cos(A) * tan(B) * (exp(B) + sin(A) * cos(B))
    @. res = res / (exp(A) + sin(B) + cos(A) + tan(B) + 1)
end

CUDA.synchronize()
