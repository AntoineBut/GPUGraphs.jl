using GPUGraphs
using BenchmarkTools
using SparseArrays
import SuiteSparseGraphBLAS: GBMatrix, GBVector, gbrand, gbset, mul!
using Metal
using KernelAbstractions
import LinearAlgebra: mul!
using DataFrames
using CSV


SUITE = BenchmarkGroup()
# Write your benchmarks here.
BACKEND = Metal.MetalBackend()  # our personal laptops
# Get number of CPU threads
n_cpu_threads = Sys.CPU_THREADS
gbset(:nthreads, n_cpu_threads)

SIZES = [16384 * 2^i for i = 1:8]
NNZ = SIZES .* 10
NB_ELEMS = 10 # Average number of non-zero elements per column
MUL_RESULTS =
    DataFrame(operation = String[], size = Int[], implementation = String[], time = Real[])

for SIZE in SIZES
    FILL = NB_ELEMS / SIZE
    print("Generating random sparse matrix of size $SIZE x $SIZE with fill $FILL\n")
    A_csc_cpu = sprand(Float32, SIZE, SIZE, FILL)
    A_csr_cpu = transpose(A_csc_cpu)
    print("Building GB sparse matrix\n")
    A_ssGB = gbrand(Float32, SIZE, SIZE, FILL)
    print("Converting to GPU format\n")
    A_csr_gpu = SparseGPUMatrixCSR(A_csr_cpu, BACKEND)
    print("Generating random vector of size $SIZE\n")
    b = rand(Float32, SIZE)
    b_ssGB = b
    b_gpu = MtlVector(b)

    semiring = Semiring(*, +, zero(Int32), one(Int32))

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
    SUITE["mul!"]["CPU"]["SuiteSparseGraphBLAS"] = @benchmarkable begin
        for i = 1:10
            mul!(res_ssGB, $A_ssGB, $b_ssGB)
        end
    end evals = 1 setup = (res_ssGB = GBVector(zeros(Float32, $SIZE)))

    SUITE["mul!"]["GPU"]["GPUGraphs"] = @benchmarkable begin
        for i = 1:10
            GPU_spmul!(res_gpu, $A_csr_gpu, $b_gpu, $semiring)
        end
        KernelAbstractions.synchronize(BACKEND)
    end evals = 1 setup = (res_gpu = allocate(BACKEND, Float32, $SIZE))

    print("Launching benchmarks\n")
    bench_res = run(SUITE)
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
        ("spmv!", SIZE, "GPUGraphs", median(bench_res["mul!"]["GPU"]["GPUGraphs"].times)),
    )

end
print(MUL_RESULTS)

# Save results to a file

CSV.write("benchmark/out/spmv_results.csv", MUL_RESULTS)
