using GPUGraphs
using Test
using KernelAbstractions
using GPUArrays
using SparseArrays
using Random
using JuliaFormatter
using Aqua
using JET
using Pkg

# Test the SparseGPUMatrixCSR utilities

# Set random seed
Random.seed!(1234)

@testset "GPUGraphs.jl" begin
    # Write your tests here.
    @test true

    @testset "Code Quality" begin
        @testset "Aqua" begin
            Aqua.test_all(GPUGraphs; ambiguities = false)
        end
        @testset "JET" begin
            #JET.test_package(GPUGraphs; target_defined_modules = true)
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(GPUGraphs; overwrite = false)
        end
    end

    TEST_BACKEND = if get(ENV, "CI", "false") == "false"
        Pkg.add("Metal")
        using Metal
        Metal.MetalBackend()  # our personal laptops
    #KernelAbstractions.CPU()
    else
        KernelAbstractions.CPU()
    end


    @testset "SparseGPUMatrixCSR" begin
        # Test the constructor
        @testset "constructor" begin
            TEST_VECTOR_TYPE_VALS = typeof(allocate(TEST_BACKEND, Float32, 0))
            TEST_VECTOR_TYPE_INDS = typeof(allocate(TEST_BACKEND, Int32, 0))
            function test_vector_types(A::SparseGPUMatrixCSR, vals, inds)
                @test typeof(A.rowptr) == inds
                @test typeof(A.colval) == inds
                @test typeof(A.nzval) == vals
            end
            @testset "empty" begin
                A = SparseGPUMatrixCSR(Float32, Int32, TEST_BACKEND)
                @test size(A, 1) == 0
                @test size(A, 2) == 0
                @test length(A) == 0
                @test nnz(A) == 0
                @test size(A) == (0, 0)
                test_vector_types(A, TEST_VECTOR_TYPE_VALS, TEST_VECTOR_TYPE_INDS)
            end

            @testset "non-empty" begin
                A_csc = sprand(Float32, 10, 10, 0.5)
                A_csc = convert(SparseMatrixCSC{Float32,Int32}, A_csc)
                A_nnz = nnz(A_csc)
                A_csr_t = sparse(transpose(A_csc))
                ref_rowptr = A_csr_t.colptr
                # Convert to int 32
                ref_colval = A_csr_t.rowval
                ref_nzval = A_csr_t.nzval

                B_0 = SparseGPUMatrixCSR(
                    10,
                    10,
                    ref_rowptr,
                    ref_colval,
                    ref_nzval,
                    TEST_BACKEND,
                )
                B_1 = SparseGPUMatrixCSR(A_csc, TEST_BACKEND)
                B_2 = SparseGPUMatrixCSR(transpose(A_csr_t), TEST_BACKEND)
                B_3 = SparseGPUMatrixCSR(collect(A_csc), TEST_BACKEND)

                function test_constructor(A::SparseGPUMatrixCSR)
                    @test size(A, 1) == 10
                    @test size(A, 2) == 10
                    @test size(A) == (10, 10)
                    @test length(A) == 100
                    @test nnz(A) == A_nnz
                    @test get_backend(A) == TEST_BACKEND
                    display(A)
                    @allowscalar @test A.rowptr == ref_rowptr
                    @allowscalar @test A.colval == ref_colval
                    @allowscalar @test A.nzval == ref_nzval
                    test_vector_types(A, TEST_VECTOR_TYPE_VALS, TEST_VECTOR_TYPE_INDS)
                    all_equal = true
                    for i = 1:10
                        for j = 1:10
                            @allowscalar all_equal = all_equal && A[i, j] == A_csc[i, j]
                        end
                    end
                    @test all_equal
                    @test_throws BoundsError A[11, 1]
                end
                test_constructor(B_0)
                test_constructor(B_1)
                test_constructor(B_2)
                test_constructor(B_3)

                A_rand = sprand_gpu(Float32, 10, 10, 0.5, TEST_BACKEND)
                @test size(A_rand, 1) == 10
                @test size(A_rand, 2) == 10
                @test size(A_rand) == (10, 10)
                @test length(A_rand) == 100

            end
            @testset "Throws" begin
                # Size mismatch
                row_ptr = [1, 4, 7, 11]
                col_val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                nz_val_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]
                nz_val_2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                @test_throws ArgumentError SparseGPUMatrixCSR(
                    3,
                    10,
                    row_ptr,
                    col_val,
                    nz_val_1,
                    TEST_BACKEND,
                )
                @test_throws ArgumentError SparseGPUMatrixCSR(
                    3,
                    10,
                    row_ptr,
                    col_val,
                    nz_val_2,
                    TEST_BACKEND,
                )

                # Rowptr mismatch
                nz_val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                @test_throws ArgumentError SparseGPUMatrixCSR(
                    4,
                    10,
                    row_ptr,
                    col_val,
                    nz_val,
                    TEST_BACKEND,
                )
                @test_throws ArgumentError SparseGPUMatrixCSR(
                    2,
                    10,
                    row_ptr,
                    col_val,
                    nz_val,
                    TEST_BACKEND,
                )
                row_ptr = [-1, 4, 7, 11]
                @test_throws ArgumentError SparseGPUMatrixCSR(
                    3,
                    10,
                    row_ptr,
                    col_val,
                    nz_val,
                    TEST_BACKEND,
                )
                row_ptr = [1, 4, 7, 12]
                @test_throws ArgumentError SparseGPUMatrixCSR(
                    3,
                    10,
                    row_ptr,
                    col_val,
                    nz_val,
                    TEST_BACKEND,
                )

                # Colval mismatch
                col_val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15]
                @test_throws ArgumentError SparseGPUMatrixCSR(
                    3,
                    10,
                    row_ptr,
                    col_val,
                    nz_val,
                    TEST_BACKEND,
                )

            end

        end
        @testset "Basic Utilities" begin
            A_cpu = sprand(Float32, 10, 10, 0.5)
            A_gpu = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
            @test size(A_gpu, 1) == 10
            @test size(A_gpu, 2) == 10
            @test size(A_gpu) == (10, 10)
            @test length(A_gpu) == 100
            @test nnz(A_gpu) == nnz(A_cpu)
        end
    end

    @testset "SparseGPUMatrixELL" begin
        @testset "constructor" begin
            TEST_VECTOR_TYPE_VALS = typeof(allocate(TEST_BACKEND, Float32, 0))
            TEST_VECTOR_TYPE_INDS = typeof(allocate(TEST_BACKEND, Int32, 0))
            function test_vector_types(A::SparseGPUMatrixELL, vals, inds)
                @test typeof(A.nnz_per_row) == inds
                @test typeof(A.colval) == inds
                @test typeof(A.nzval) == vals
            end

            @testset "non-empty" begin

                A_dense = [
                    1 7 0 0
                    5 0 3 9
                    0 2 8 0
                    0 0 0 6
                ]
                A_csc = convert(SparseMatrixCSC{Float32,Int32}, A_dense)
                A_csc_t = convert(SparseMatrixCSC{Float32,Int32}, transpose(A_dense))

                A_nnz = 8
                # Convert
                A_dense = convert(Matrix{Float32}, A_dense)
                ref_nnz_per_row = [2, 3, 2, 1]
                ref_colval = [1, 1, 2, 4, 2, 3, 3, 0, 0, 4, 0, 0]
                ref_nzval = [1, 5, 2, 6, 7, 3, 8, 0, 0, 9, 0, 0]

                B_1 = SparseGPUMatrixELL(A_csc, TEST_BACKEND)
                B_2 = SparseGPUMatrixELL(transpose(A_csc_t), TEST_BACKEND)
                B_3 = SparseGPUMatrixELL(A_dense, TEST_BACKEND)

                function test_constructor(A::SparseGPUMatrixELL)
                    @test size(A, 1) == 4
                    @test size(A, 2) == 4
                    @test size(A) == (4, 4)
                    @test length(A) == 16
                    @test nnz(A) == A_nnz
                    @test get_backend(A) == TEST_BACKEND
                    display(A)
                    @allowscalar @test A.nnz_per_row == ref_nnz_per_row
                    @allowscalar @test A.colval == ref_colval
                    @allowscalar @test A.nzval == ref_nzval
                    test_vector_types(A, TEST_VECTOR_TYPE_VALS, TEST_VECTOR_TYPE_INDS)
                    all_equal = true
                    for i = 1:4
                        for j = 1:4
                            @allowscalar all_equal = all_equal && A[i, j] == A_dense[i, j]
                        end
                    end
                    @test all_equal
                    @test_throws BoundsError A[11, 1]
                end
                test_constructor(B_1)
                test_constructor(B_2)
                test_constructor(B_3)
            end
        end

    end

    @testset "SparseGPUVector" begin
        @testset "Constructors" begin
            @testset "empty" begin
                A = SparseGPUVector(Float32, Int32, TEST_BACKEND)
                @test size(A) == 0
                @test length(A) == 0
            end

            @testset "non-empty" begin
                A_cpu = sprand(Float32, 10, 0.5)
                ref_nzind = A_cpu.nzind
                ref_nzval = A_cpu.nzval

                B_0 = SparseGPUVector(10, ref_nzind, ref_nzval, TEST_BACKEND)
                B_1 = SparseGPUVector(A_cpu, TEST_BACKEND)
                B_2 = SparseGPUVector(collect(A_cpu), TEST_BACKEND)

                test_vectors = [B_0, B_1, B_2]
                i = 1
                for B in test_vectors
                    i += 1
                    @test size(B) == 10
                    @test length(B) == 10
                    @allowscalar @test B.nzind == ref_nzind
                    @allowscalar @test B.nzval == ref_nzval
                end
            end
        end
    end
    @testset "GraphBLAS" begin

        @testset "mul!" begin
            # Matrix-vector multiplication
            A_cpu = sprand(Float32, 10, 10, 0.5)
            B_cpu = rand(Float32, 10)
            C_cpu = A_cpu * B_cpu
            A_gpu = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
            B_gpu = allocate(TEST_BACKEND, Float32, 10)
            copyto!(B_gpu, B_cpu)
            C_gpu = KernelAbstractions.zeros(TEST_BACKEND, Float32, 10)
            #semiring = Semiring(*, Monoid(+, 0.0), 0.0, 1.0)

            gpu_spmv!(C_gpu, A_gpu, B_gpu)
            @allowscalar @test C_gpu == C_cpu

            # Large matrix
            LARGE_NB = 1000
            A_cpu = sprand(Float32, LARGE_NB, LARGE_NB, 0.2)
            B_cpu = rand(Float32, LARGE_NB)
            C_cpu = A_cpu * B_cpu
            A_gpu = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
            B_gpu = allocate(TEST_BACKEND, Float32, LARGE_NB)
            copyto!(B_gpu, B_cpu)
            C_gpu = KernelAbstractions.zeros(TEST_BACKEND, Float32, LARGE_NB)
            #semiring = Semiring((x, y) -> x * y, Monoid(+, 0.0), 0.0, 1.0)

            gpu_spmv!(C_gpu, A_gpu, B_gpu)
            KernelAbstractions.synchronize(TEST_BACKEND)

            #@allowscalar @test C_gpu == C_cpu
            # Count the number of differences
            diff = 0
            for i = 1:LARGE_NB
                if @allowscalar abs(C_gpu[i] - C_cpu[i]) > 1e-6
                    diff += 1
                end
            end
            @test diff <= 0
            println("Number of differences: $diff out of $LARGE_NB")

        end

    end
end
