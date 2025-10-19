using GPUGraphs
using Test
using KernelAbstractions
using LinearAlgebra
using GPUArrays
using SparseArrays
using Random
using JuliaFormatter
using Aqua
using JET
using Pkg
using Graphs
using SimpleWeightedGraphs
using CUDA

# Test the SparseGPUMatrixCSR utilities

# Set random seed
Random.seed!(1234)

# Value used in Colval for padding zero entries
const PAD_VAL = -1
# We use 1 instead of 0 so that during spmv(A, b), attempting to do b[colval[i]] * nzval[i] on a structural zero yields b[colval[1]] * 0 = 0
# instead of b[colval[0]] * 0 => BoundsError (or worse, no error but silently wrong result)

@testset "GPUGraphs.jl" begin
    # Write your tests here.
    @test true

    @testset "Code Quality" begin
        @testset "Aqua" begin
            #Aqua.test_all(GPUGraphs; ambiguities = false)
        end
        @testset "JET" begin
            #JET.test_package(GPUGraphs; target_defined_modules = true)
        end
        @testset "JuliaFormatter" begin
            #@test JuliaFormatter.format(GPUGraphs; overwrite = false)
        end
    end



    include("structs.jl")
    include("spmv.jl")
    include("spmm.jl")
    include("bfs.jl")
    include("shortest_path.jl")
end
