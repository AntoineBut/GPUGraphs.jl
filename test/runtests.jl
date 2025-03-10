using GPUGraphs
using Test
using KernelAbstractions
using GPUArrays
using SparseArrays
using Random
using Metal
using JuliaFormatter
using Aqua
using JET

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
            JET.test_package(GPUGraphs; target_defined_modules = true)
        end
        @testset "JuliaFormatter" begin
            @test JuliaFormatter.format(GPUGraphs; overwrite = false)
        end
    end



    TEST_BACKEND = Metal.MetalBackend() # TODO : change this to select the backend automatically
    @testset "SparseGPUMatrixCSR Utilities" begin

        # Test the constructor
        @testset "constructor" begin
            @testset "empty" begin
                A = SparseGPUMatrixCSR(Float32, Int32, TEST_BACKEND)
                @test size(A, 1) == 0
                @test size(A, 2) == 0
                @test length(A) == 0
            end

            @testset "non-empty" begin
                A_csc = sprand(Float32, 10, 10, 0.5)
                A_nnz = nnz(A_csc)
                A_csr_t = sparse(transpose(A_csc))
                ref_rowptr = A_csr_t.colptr
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

                test_matrices = [B_0, B_1, B_2, B_3]
                i = 0
                for B in test_matrices
                    println("##### Test Constructor ", i)
                    i += 1
                    @test size(B, 1) == 10
                    @test size(B, 2) == 10
                    @test length(B) == 100
                    @test nnz(B) == A_nnz
                    @allowscalar @test B.rowptr == ref_rowptr
                    @allowscalar @test B.colval == ref_colval
                    @allowscalar @test B.nzval == ref_nzval
                end
            end

        end
        @testset "Basic Utilities" begin
            A = SparseGPUMatrixCSR(Float32, Int32, TEST_BACKEND)
            @test size(A, 1) == 0
            @test size(A, 2) == 0
            @test length(A) == 0
            @test nnz(A) == 0

            A_cpu = sprand(Float32, 10, 10, 0.5)
            A_gpu = SparseGPUMatrixCSR(A_cpu, TEST_BACKEND)
            @test size(A_gpu, 1) == 10
            @test size(A_gpu, 2) == 10
            @test length(A_gpu) == 100
            @test nnz(A_gpu) == nnz(A_cpu)
        end
    end
end
