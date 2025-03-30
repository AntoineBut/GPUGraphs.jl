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
MAIN_TYPE = Bool
INDEX_TYPE = Int32

ELTYPE_A = MAIN_TYPE
ELTYPE_B = MAIN_TYPE
ELTYPE_RES = MAIN_TYPE

MUL = *
ADD = +
ACCUM = +

if MAIN_TYPE == Bool
    MUL = &
    ADD = |
    ACCUM = |
    SEMIRING = semiring(∧, ∨, Bool, Bool)
end


# Write your benchmarks here.
BACKEND = Metal.MetalBackend()  # our personal laptops
# Get number of CPU threads
n_cpu_threads = Sys.CPU_THREADS
gbset(:nthreads, n_cpu_threads)

SIZES = [16384 * 2^i for i = 1:1]
NNZ = SIZES .* 10
NB_ELEMS = 10 # Average number of non-zero elements per column
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

    """
        SUITE["mul!"]["CPU"]["SparseArrays-CSR"] = @benchmarkable begin
            for i = 1:10
                mul!(res, $A_csr_cpu, $b)
            end
        end evals = 1 setup = (res = zeros(Float32, $SIZE))

        SUITE["mul!"]["CPU"]["SparseArrays-CSC"] = @benchmarkable begin
            for i = 1:10
                mul!(res, $A_csc_cpu, $b)
            end
        end evals = 1 setup = (res = zeros(Float32, $SIZE))

        """
    SUITE["mul!"]["CPU"]["SuiteSparseGraphBLAS"] = @benchmarkable begin
        for i = 1:10
            mul!(res_ssGB, $A_ssGB, $b_ssGB, (∧, ∨); accum = ∨)
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
    """
        SUITE["mul!"]["GPU"]["GPUGraphsELLGrouped"] = @benchmarkable begin
            for i = 1:10
                gpu_spmv!(res_gpu, $A_ell_gpu, $b_gpu, 256)
            end
            KernelAbstractions.synchronize(BACKEND)
        end evals = 1 setup = (res_gpu = KernelAbstractions.zeros(BACKEND, Float32, $SIZE))
    """
    print("Launching benchmarks\n")
    bench_res = run(SUITE)
    """
    push!(
        MUL_RESULTS,
        (
            "spmv!",
            SIZE,
            "SparseArrays-CSR",
            median(bench_res["mul!"]["CPU"]["SparseArrays-CSR"].times),
        ),
    )
    push!(
        MUL_RESULTS,
        (
            "spmv!",
            SIZE,
            "SparseArrays-CSC",
            median(bench_res["mul!"]["CPU"]["SparseArrays-CSC"].times),
        ),
    )
        """
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
    """
    push!(
        MUL_RESULTS,
        (
            "spmv!",
            SIZE,
            "GPUGraphsELLGrouped",
            median(bench_res["mul!"]["GPU"]["GPUGraphsELLGrouped"].times),
        ),
    )
        """

end
println(MUL_RESULTS)

# Save results to a file

CSV.write("benchmark/out/spmv_results.csv", MUL_RESULTS)

println("Loading graph data...")
# Load dataset

A_T = adjacency_matrix(
    loadgraph("benchmark/data/italy_osm/italy_osm.mtx", EdgeListFormat()),
    MAIN_TYPE;
    dir = :in,
)
println("Done. ")

SIZE = size(A_T, 1)
A_csr_gpu = SparseGPUMatrixCSR(transpose(A_T), Metal.MetalBackend())
A_ssGB = GBMatrix{MAIN_TYPE}(A_T)
A_ell_gpu = SparseGPUMatrixELL(transpose(A_T), Metal.MetalBackend())

b = rand(MAIN_TYPE, SIZE)
b_ssGB = b
b_gpu = MtlVector(b)

SUITE2 = BenchmarkGroup()


SUITE2["mul!"]["CPU"]["SuiteSparseGraphBLAS"] = @benchmarkable begin
    for i = 1:10
        mul!(res_ssGB, $A_ssGB, $b_ssGB, (∧, ∨); accum = ∨)
    end
end evals = 1 setup = (res_ssGB = GBVector(zeros(ELTYPE_RES, $SIZE)))

SUITE2["mul!"]["GPU"]["GPUGraphsCSR"] = @benchmarkable begin
    for i = 1:10
        gpu_spmv!(res_gpu, $A_csr_gpu, $b_gpu, MUL, ADD, ACCUM)
    end
    KernelAbstractions.synchronize(BACKEND)
end evals = 1 setup = (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))


SUITE2["mul!"]["GPU"]["GPUGraphsELL"] = @benchmarkable begin
    for i = 1:10
        gpu_spmv!(res_gpu, A_ell_gpu, b_gpu, MUL, ADD, ACCUM)
    end
    KernelAbstractions.synchronize(BACKEND)
end evals = 1 setup = (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))


println("Launching benchmarks\n")
bench_res2 = run(SUITE2)


DATA_RESULTS = DataFrame(
    operation = String[],
    dataset = String[],
    implementation = String[],
    time = Real[],
)

push!(
    DATA_RESULTS,
    (
        "spmv!",
        "com-Orkut",
        "SuiteSparseGraphBLAS",
        median(bench_res2["mul!"]["CPU"]["SuiteSparseGraphBLAS"].times),
    ),
)
push!(
    DATA_RESULTS,
    (
        "spmv!",
        "com-Orkut",
        "GPUGraphsCSR",
        median(bench_res2["mul!"]["GPU"]["GPUGraphsCSR"].times),
    ),
)
push!(
    DATA_RESULTS,
    (
        "spmv!",
        "com-Orkut",
        "GPUGraphsELL",
        median(bench_res2["mul!"]["GPU"]["GPUGraphsELL"].times),
    ),
)
println(DATA_RESULTS)
CSV.write("benchmark/out/spmv_results_osm.csv", DATA_RESULTS)
println("Done. ")
