using GPUGraphs
using BenchmarkTools
using SparseArrays
import SuiteSparseGraphBLAS: GBMatrix, GBVector, gbrand, gbset, mul!, semiring, Monoid, ∧, ∨
#using Metal
using CUDA
using KernelAbstractions
import LinearAlgebra: mul!
using DataFrames
using CSV
using Graphs
using GraphIO.EdgeList
using SuiteSparseMatrixCollection
using HarwellRutherfordBoeing
using ParallelGraphs


SUITE = BenchmarkGroup()
MAIN_TYPE = Float32
BOOL_TYPE = Bool
INDEX_TYPE = Int32
INDEX_SSGB = Int

ELTYPE_A = MAIN_TYPE
ELTYPE_B = MAIN_TYPE
ELTYPE_RES = MAIN_TYPE

MUL = GPUGraphs_mul
ADD = GPUGraphs_add
ACCUM = GPUGraphs_second
ACCUM_SSGB = +
SEMIRING = semiring(+,*,MAIN_TYPE,MAIN_TYPE)

if MAIN_TYPE == Bool
    MUL = GPUGraphs_band
    ADD = GPUGraphs_bor
    ACCUM = GPUGraphs_second
    ACCUM_SSGB = ∨
    SEMIRING = semiring(∨,∧,MAIN_TYPE,MAIN_TYPE)
end


# Write your benchmarks here.
#BACKEND = Metal.MetalBackend()  # our personal laptops
BACKEND = CUDA.CUDABackend()  # on the cluster
# Get number of CPU threads
n_cpu_threads = Sys.CPU_THREADS
SIZES = [16384 * 2^i for i = 0:5]
#SIZES = []
NB_ELEMS = 20 # Average number of non-zero elements per column
NNZ = SIZES .* NB_ELEMS
MUL_RESULTS =
    DataFrame(operation = String[], size = Int[], implementation = String[], time = Real[])

BFS_RESULTS =
    DataFrame(operation = String[], size = Int[], implementation = String[], time = Real[])

for SIZE in SIZES
    SIZE2 = SIZE
    FILL = 0.001
    print("Generating random sparse matrix of size $SIZE x $SIZE2 with fill $FILL\n")
    A_csc_cpu =
        convert(SparseMatrixCSC{MAIN_TYPE,INDEX_TYPE}, sprand(MAIN_TYPE, SIZE2, SIZE, FILL))
    A_csr_cpu = transpose(A_csc_cpu)
    print("Converting to GPU format\n")
    A_csr_gpu = SparseGPUMatrixCSR(A_csr_cpu, BACKEND)
    A_sell_gpu = SparseGPUMatrixSELL(A_csr_cpu, BACKEND)
    print("Generating random vector of size $SIZE\n")
    b = rand(ELTYPE_B, SIZE)
    b_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_B, SIZE)
    copyto!(b_gpu, b)
    print("Building GB sparse matrix\n")
    A_ssGB = gbrand(ELTYPE_A, SIZE, SIZE2, FILL)
    b_ssGB = b


    SUITE["mul!"]["CPU"]["JuliaSparse"] = @benchmarkable begin
        for i = 1:10
            mul!(res, $A_csr_cpu, $b)
        end
    end evals = 1 setup =
        (res_ssGB = GBVector{INDEX_TYPE}($SIZE, fill = zero(INDEX_TYPE)))

    SUITE["mul!"]["GPU"]["GPUGraphsCSR"] = @benchmarkable begin
        for i = 1:10
            gpu_spmv!(res_gpu, $A_csr_gpu, $b_gpu; mul = MUL, add = ADD, accum = ACCUM)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))

    SUITE["mul!"]["GPU"]["GPUGraphsSELL"] = @benchmarkable begin
        for i = 1:10
            gpu_spmv!(res_gpu, $A_sell_gpu, $b_gpu; mul = MUL, add = ADD, accum = ACCUM)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))

    if MAIN_TYPE == Bool
        # BFS for bool
        GSIZE = SIZE * 16
        print("Generating random graph of size $GSIZE \n")
        graph = dorogovtsev_mendes(GSIZE)
        A_csc_cpu = convert(
            SparseMatrixCSC{MAIN_TYPE,INDEX_TYPE},
            adjacency_matrix(graph, MAIN_TYPE; dir = :out),
        )
        A_csr_cpu = transpose(A_csc_cpu)
        print("Converting to GPU-CSR format\n")
        A_csr_gpu = SparseGPUMatrixCSR(A_csr_cpu, BACKEND)
        print("Converting to SELL format\n")
        A_sell_gpu = SparseGPUMatrixSELL(A_csr_cpu, BACKEND)

        print("Building GB sparse matrix\n")
        A_ssGB = GBMatrix(
            convert(
                SparseMatrixCSC{MAIN_TYPE,INDEX_SSGB},
                adjacency_matrix(graph, Bool; dir = :out),
            ),
        )

        SUITE["bfs"]["CPU"]["Graphs.jl"] = @benchmarkable begin
            Graphs.bfs_parents($graph, one(INDEX_TYPE))
        end evals = 1

        SUITE["bfs"]["GPU"]["GPUGraphsCSR"] = @benchmarkable begin
            GPUGraphs.bfs_parents($A_csr_gpu, one(INDEX_TYPE))
            KernelAbstractions.synchronize(BACKEND)
        end evals = 1

        SUITE["bfs"]["GPU"]["GPUGraphsSELL"] = @benchmarkable begin
            GPUGraphs.bfs_parents($A_sell_gpu, one(INDEX_TYPE))
            KernelAbstractions.synchronize(BACKEND)
        end evals = 1

        SUITE["bfs"]["CPU"]["SuiteSparseGraphBLAS"] = @benchmarkable begin
            bfs_BLAS!($A_ssGB, one(INDEX_TYPE), res_ssGB)
        end evals = 1 setup =
            (res_ssGB = GBVector{INDEX_TYPE}($GSIZE, fill = zero(INDEX_TYPE)))
    end


    println("Launching benchmarks\n")
    bench_res = run(SUITE)

    push!(
        MUL_RESULTS,
        (
            "spmv!",
            SIZE,
            "JuliaSparse",
            median(bench_res["mul!"]["CPU"]["JuliaSparse"].times),
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
            "GPUGraphsSELL",
            median(bench_res["mul!"]["GPU"]["GPUGraphsSELL"].times),
        ),
    )

    if MAIN_TYPE == Bool
        push!(
            BFS_RESULTS,
            ("bfs", SIZE, "Graphs.jl", median(bench_res["bfs"]["CPU"]["Graphs.jl"].times)),
        )
        push!(
            BFS_RESULTS,
            (
                "bfs",
                SIZE,
                "GPUGraphsCSR",
                median(bench_res["bfs"]["GPU"]["GPUGraphsCSR"].times),
            ),
        )
        push!(
            BFS_RESULTS,
            (
                "bfs",
                SIZE,
                "GPUGraphsSELL",
                median(bench_res["bfs"]["GPU"]["GPUGraphsSELL"].times),
            ),
        )
        push!(
            BFS_RESULTS,
            (
                "bfs",
                SIZE,
                "SuiteSparseGraphBLAS",
                median(bench_res["bfs"]["CPU"]["SuiteSparseGraphBLAS"].times),
            ),
        )
    end
    println("Done with size $SIZE")
end
println("Results for synthetic data")
println(MUL_RESULTS)
println("------")
println(BFS_RESULTS)
println("Done. ")


# Save results to a file

CSV.write("benchmark/out/spmv_results.csv", MUL_RESULTS)
CSV.write("benchmark/out/bfs_results.csv", BFS_RESULTS)
# Thow error to stop execution
error("Stopping execution after synthetic benchmarks.")

NB_DATASETS = 3

ssmc = ssmc_db()
orkut_path = fetch_ssmc(ssmc_matrices(ssmc, "SNAP", "Orkut"), format = "RB")
#live_journal_path = fetch_ssmc(ssmc_matrices(ssmc, "SNAP", "com-LiveJournal"), format="RB")
osm_path = fetch_ssmc(ssmc_matrices(ssmc, "DIMACS10", "italy_osm"), format = "RB")
nlpkkt_path = fetch_ssmc(ssmc_matrices(ssmc, "Schenk", "nlpkkt160"), format = "RB")

DATASET_NAMES = ["com-Orkut", "italy_osm", "nlpkkt160"]
DATASET_PATHS = [
    orkut_path[1],
    #live_journal_path[1],
    osm_path[1],
    nlpkkt_path[1],
]

DATA_MUL_RESULTS = DataFrame(
    operation = String[],
    dataset = String[],
    implementation = String[],
    time = Real[],
)

DATA_BFS_RESULTS = DataFrame(
    operation = String[],
    dataset = String[],
    implementation = String[],
    time = Real[],
)

for i = 1:4
    if i == 2
        continue # Skip LiveJournal
    end

    println("Loading graph data for dataset $(DATASET_NAMES[i])")
    # Load dataset
    loaded_matrix =
        RutherfordBoeingData(joinpath(DATASET_PATHS[i], "$(DATASET_NAMES[i]).rb"))
    println("Loaded. ")

    if i == 3 && MAIN_TYPE <: Integer
        # nlpkkt160 is a float matrix, we need to convert it to a bool matrix 
        loaded_matrix.data.nzval .= 1.0
    end

    A_T = adjacency_matrix(SimpleDiGraph(loaded_matrix.data), Bool; dir = :both)
    A_T = convert(SparseMatrixCSC{MAIN_TYPE,INDEX_TYPE}, A_T)
    println("Converted to CSC.")

    SIZE = size(A_T, 1)
    A_sell_gpu = SparseGPUMatrixSELL(transpose(A_T), 32, BACKEND)
    println("Converted to GPU-SELL.")
    A_csr_gpu = SparseGPUMatrixCSR(transpose(A_T), BACKEND)
    println("Converted to GPU-CSR.")

    A_ssGB = GBMatrix(convert(SparseMatrixCSC{MAIN_TYPE,INDEX_SSGB}, A_T))
    println("Converted to GBMatrix.")

    cusparse_A_csr = CUSPARSE.CuSparseMatrixCSR(A_T)
    println("Converted to CUSPARSE-CSR.")

    if i != 1
        A_sell_gpu = SparseGPUMatrixSELL(transpose(A_T), BACKEND)
    end
    b = rand(MAIN_TYPE, SIZE)
    b_ssGB = b
    b_gpu = KernelAbstractions.allocate(BACKEND, MAIN_TYPE, SIZE)
    copyto!(b_gpu, b)
    println("Creating benchmark groups\n")
    SUITE2 = BenchmarkGroup()

    SUITE2["mul!"]["CPU"]["JuliaSparse"] = @benchmarkable begin
        for i = 1:10
            mul!(res, $A_csr_cpu, $b)
        end
    end evals = 1 setup =
        (res_ssGB = GBVector{INDEX_TYPE}($SIZE, fill = zero(INDEX_TYPE)))

    SUITE2["mul!"]["GPU"]["GPUGraphsCSR"] = @benchmarkable begin
        for j = 1:10
            gpu_spmv!(res_gpu, $A_csr_gpu, $b_gpu; mul = MUL, add = ADD, accum = ACCUM)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))

    SUITE2["mul!"]["GPU"]["GPUGraphsSELL"] = @benchmarkable begin
        for j = 1:10
            gpu_spmv!(res_gpu, $A_sell_gpu, $b_gpu; mul = MUL, add = ADD, accum = ACCUM)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))

    SUITE2["mul!"]["GPU"]["CUSPARSE-CSR"] = @benchmarkable begin
        for j = 1:10
            mul!(res_gpu, $cusparse_A_csr, $b_gpu)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))


    graph = SimpleGraph(A_T)
    println("Built graph. ")

    A_csr_cpu = transpose(
        convert(
            SparseMatrixCSC{BOOL_TYPE,INDEX_TYPE},
            adjacency_matrix(graph, BOOL_TYPE; dir = :out),
        ),
    )
    print("Converting to GPU-CSR format\n")
    A_csr_gpu = SparseGPUMatrixCSR(A_csr_cpu, BACKEND)

    print("Converting to SELL format\n")
    A_sell_gpu = SparseGPUMatrixSELL(A_csr_cpu, 64, BACKEND)

    SUITE2["bfs"]["CPU"]["SuiteSparseGraphBLAS"] = @benchmarkable begin
        bfs_BLAS!($A_ssGB, one(INDEX_TYPE), res_ssGB)
    end evals = 1 setup =
        (res_ssGB = GBVector{INDEX_TYPE}($SIZE, fill = zero(INDEX_TYPE)))

    SUITE2["bfs"]["GPU"]["GPUGraphsCSR"] = @benchmarkable begin
        GPUGraphs.bfs_parents($A_csr_gpu, one(INDEX_TYPE))
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1

end

if i >= 2

    SUITE2["mul!"]["GPU"]["GPUGraphsSELL"] = @benchmarkable begin
        for j = 1:10
            gpu_spmv!(res_gpu, $A_sell_gpu, $b_gpu; mul = MUL, add = ADD, accum = ACCUM)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup =
        (res_gpu = KernelAbstractions.zeros(BACKEND, ELTYPE_RES, $SIZE))

    if MAIN_TYPE == Bool
        SUITE2["bfs"]["GPU"]["GPUGraphsSELL"] = @benchmarkable begin
            GPUGraphs.bfs_parents($A_sell_gpu, one(INDEX_TYPE))
            KernelAbstractions.synchronize(BACKEND)
        end evals = 1
    end

    println("Launching benchmarks\n")
    bench_res2 = run(SUITE2)
    println("Done. ")

    ## Graphs.jl ##

    push!(
        DATA_BFS_RESULTS,
        (
            "bfs",
            DATASET_NAMES[i],
            "Graphs.jl",
            median(bench_res2["bfs"]["CPU"]["Graphs.jl"].times),
        ),
    )

    ## CPU ## 
    push!(
        DATA_MUL_RESULTS,
        (
            "spmv!",
            DATASET_NAMES[i],
            "JuliaSparse",
            median(bench_res2["mul!"]["CPU"]["JuliaSparse"].times),
        ),
    )
    push!(
        DATA_BFS_RESULTS,
        (
            "bfs",
            DATASET_NAMES[i],
            "SuiteSparseGraphBLAS",
            median(bench_res2["bfs"]["CPU"]["SuiteSparseGraphBLAS"].times),
        ),
    )
    if MAIN_TYPE == Bool
        push!(
            DATA_BFS_RESULTS,
            (
                "bfs",
                DATASET_NAMES[i],
                "SuiteSparseGraphBLAS",
                median(bench_res2["bfs"]["CPU"]["SuiteSparseGraphBLAS"].times),
            ),
        )
    end

    ## GPU - CSR ##
    push!(
        DATA_MUL_RESULTS,
        (
            "spmv!",
            DATASET_NAMES[i],
            "GPUGraphsCSR",
            median(bench_res2["mul!"]["GPU"]["GPUGraphsCSR"].times),
        ),
    )
    push!(
        DATA_BFS_RESULTS,
        (
            "bfs",
            DATASET_NAMES[i],
            "GPUGraphsCSR",
            median(bench_res2["bfs"]["GPU"]["GPUGraphsCSR"].times),
        ),
    )

    ## GPU - ELL ## (skiped for Orkut and LiveJournal)
    if i >= 2

        push!(
            DATA_MUL_RESULTS,
            (
                "spmv!",
                DATASET_NAMES[i],
                "GPUGraphsSELL",
                median(bench_res2["mul!"]["GPU"]["GPUGraphsSELL"].times),
            ),
        )

        if MAIN_TYPE == Bool
            push!(
                DATA_BFS_RESULTS,
                (
                    "bfs",
                    DATASET_NAMES[i],
                    "GPUGraphsSELL",
                    median(bench_res2["bfs"]["GPU"]["GPUGraphsSELL"].times),
                ),
            )
        end

    else
        push!(DATA_MUL_RESULTS, ("spmv!", DATASET_NAMES[i], "GPUGraphsSELL", 0))
        if MAIN_TYPE == Bool
            push!(DATA_BFS_RESULTS, ("bfs", DATASET_NAMES[i], "GPUGraphsSELL", 0))
        end
    end
end
println(DATA_MUL_RESULTS)
println(DATA_BFS_RESULTS)
CSV.write("benchmark/out/spmv_results_data.csv", DATA_MUL_RESULTS)
CSV.write("benchmark/out/bfs_results_data.csv", DATA_BFS_RESULTS)
println("Done. ")
