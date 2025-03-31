using GPUGraphs
using BenchmarkTools
using SparseArrays
import SuiteSparseGraphBLAS: GBMatrix, GBVector, gbrand, gbset, mul!, semiring, Monoid, ∧, ∨
using Metal
using KernelAbstractions
import LinearAlgebra: mul!
using DataFrames
using CSV
using Graphs
using GraphIO.EdgeList


SUITE = BenchmarkGroup()
MAIN_TYPE = Int32
INDEX_TYPE = Int32

ELTYPE_A = MAIN_TYPE
ELTYPE_B = MAIN_TYPE
ELTYPE_RES = MAIN_TYPE

MUL = *
ADD = +
ACCUM = +
ACCUM_SSGB = +
SEMIRING = semiring(*, +, MAIN_TYPE, MAIN_TYPE)

if MAIN_TYPE == Bool
    MUL = &
    ADD = |
    ACCUM = |
    SEMIRING = semiring(∧, ∨, Bool, Bool)
    ACCUM_SSGB = ∨
end


# Write your benchmarks here.
BACKEND = Metal.MetalBackend()  # our personal laptops
# Get number of CPU threads
n_cpu_threads = Sys.CPU_THREADS
gbset(:nthreads, n_cpu_threads)

SIZES = [16384 * 2^i for i = 1:8]
NB_ELEMS = 40 # Average number of non-zero elements per column
NNZ = SIZES .* NB_ELEMS
MUL_RESULTS =
    DataFrame(operation = String[], size = Int[], implementation = String[], time = Real[])

for SIZE in SIZES
    FILL = NB_ELEMS / SIZE
    print("Generating random sparse matrix of size $SIZE x $SIZE with fill $FILL\n")
    A_csc_cpu = sprand(ELTYPE_A, SIZE, SIZE, FILL)
    A_csr_cpu = transpose(A_csc_cpu)
    print("Converting to GPU format\n")
    A_csr_gpu = SparseGPUMatrixCSR(A_csr_cpu, BACKEND)
    A_ell_gpu = SparseGPUMatrixELL(A_csr_cpu, BACKEND)
    print("Generating random vector of size $SIZE\n")
    b = rand(ELTYPE_B, SIZE)
    b_gpu = MtlVector(b)
    print("Building GB sparse matrix\n")
    A_ssGB = gbrand(ELTYPE_A, SIZE, SIZE, FILL)
    b_ssGB = b

    SUITE["mul!"]["CPU"]["SuiteSparseGraphBLAS"] = @benchmarkable begin
        for i = 1:10
            mul!(res_ssGB, $A_ssGB, $b_ssGB, SEMIRING; accum = ACCUM_SSGB)
        end
    end evals = 1 setup = (res_ssGB = GBVector(zeros(ELTYPE_RES, $SIZE)))

    SUITE["mul!"]["GPU"]["GPUGraphsCSR"] = @benchmarkable begin
        for i = 1:10
            gpu_spmv!(res_gpu, $A_csr_gpu, $b_gpu, MUL, ADD, ACCUM)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))

    SUITE["mul!"]["GPU"]["GPUGraphsELL"] = @benchmarkable begin
        for i = 1:10
            gpu_spmv!(res_gpu, $A_ell_gpu, $b_gpu, MUL, ADD, ACCUM)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))

    print("Launching benchmarks\n")
    bench_res = run(SUITE)

    push!(
        MUL_RESULTS,
        (
            "spmv!",
            SIZE,
            "SuiteSparseGraphBLAS",
            median(bench_res["mul!"]["CPU"]["SuiteSparseGraphBLAS"].times),
        ),
    )
    push!(
        MUL_RESULTS,
        (
            "spmv!",
            SIZE,
            "GPUGraphsCSR",
            median(bench_res["mul!"]["GPU"]["GPUGraphsCSR"].times),
        ),
    )

    push!(
        MUL_RESULTS,
        (
            "spmv!",
            SIZE,
            "GPUGraphsELL",
            median(bench_res["mul!"]["GPU"]["GPUGraphsELL"].times),
        ),
    )

end
println(MUL_RESULTS)

# Save results to a file

CSV.write("benchmark/out/spmv_results.csv", MUL_RESULTS)

NB_DATASETS = 3
DATASET_NAMES = ["OSM", "NLP-KKT", "Orkut"]
DATASET_PATHS = [
    "benchmark/data/italy_osm/italy_osm.mtx",
    "benchmark/data/nlpkkt160/nlpkkt160-bool.mtx",
    "benchmark/data/com-Orkut/com-Orkut.mtx",
]

DATA_RESULTS = DataFrame(
    operation = String[],
    dataset = String[],
    implementation = String[],
    time = Real[],
)

for i = 1:NB_DATASETS

    println("Loading graph data for dataset $(DATASET_NAMES[i])")
    # Load dataset

    A_T = adjacency_matrix(
        SimpleGraph(loadgraph(DATASET_PATHS[i], EdgeListFormat())),
        MAIN_TYPE;
        dir = :in,
    )
    println("Done. ")

    SIZE = size(A_T, 1)
    A_csr_gpu = SparseGPUMatrixCSR(transpose(A_T), Metal.MetalBackend())
    A_ssGB = GBMatrix{MAIN_TYPE}(A_T)

    if i != 3
        A_ell_gpu = SparseGPUMatrixELL(transpose(A_T), Metal.MetalBackend())
    end

    b = rand(MAIN_TYPE, SIZE)
    b_ssGB = b
    b_gpu = MtlVector(b)

    SUITE2 = BenchmarkGroup()
    SUITE2["mul!"]["CPU"]["SuiteSparseGraphBLAS"] = @benchmarkable begin
        for i = 1:10
            mul!(res_ssGB, $A_ssGB, $b_ssGB, SEMIRING; accum = ACCUM_SSGB)
        end
    end evals = 1 setup = (res_ssGB = GBVector(zeros(ELTYPE_RES, $SIZE)))

    SUITE2["mul!"]["GPU"]["GPUGraphsCSR"] = @benchmarkable begin
        for i = 1:10
            gpu_spmv!(res_gpu, $A_csr_gpu, $b_gpu, MUL, ADD, ACCUM)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))

    if i != 3

        SUITE2["mul!"]["GPU"]["GPUGraphsELL"] = @benchmarkable begin
            for i = 1:10
                gpu_spmv!(res_gpu, $A_ell_gpu, $b_gpu, MUL, ADD, ACCUM)
            end
            KernelAbstractions.synchronize(BACKEND)
        end evals = 1 setup =
            (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))
    end
    println("Launching benchmarks\n")
    bench_res2 = run(SUITE2)
    println("Done. ")
    push!(
        DATA_RESULTS,
        (
            "spmv!",
            DATASET_NAMES[i],
            "SuiteSparseGraphBLAS",
            median(bench_res2["mul!"]["CPU"]["SuiteSparseGraphBLAS"].times),
        ),
    )
    push!(
        DATA_RESULTS,
        (
            "spmv!",
            DATASET_NAMES[i],
            "GPUGraphsCSR",
            median(bench_res2["mul!"]["GPU"]["GPUGraphsCSR"].times),
        ),
    )
    if i != 3
        push!(
            DATA_RESULTS,
            (
                "spmv!",
                DATASET_NAMES[i],
                "GPUGraphsELL",
                median(bench_res2["mul!"]["GPU"]["GPUGraphsELL"].times),
            ),
        )
    else
        push!(DATA_RESULTS, ("spmv!", DATASET_NAMES[i], "GPUGraphsELL", 0))
    end
end
println(DATA_RESULTS)
CSV.write("benchmark/out/spmv_results_data.csv", DATA_RESULTS)
println("Done. ")
