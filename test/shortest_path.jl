TEST_BACKEND = if get(ENV, "CI", "false") == "false"

    #Metal.MetalBackend()  # our personal laptops
    CUDA.CUDABackend()  # on the cluster
# KernelAbstractions.CPU()
else
    KernelAbstractions.CPU()
end

g1 = barabasi_albert(30, 3)
g2 = dorogovtsev_mendes(30)
#g3 = grid((10, 10))
test_graphs = [g1, g2]

# Create random weighted graphs
g1 = barabasi_albert(30, 3)
g1_w = SimpleWeightedGraph(g1)
weights = rand(1:10, ne(g1))
for e in edges(g1)
    add_edge!(g1_w, src(e), dst(e), weights[e.src])
end
g2 = dorogovtsev_mendes(30)
g2_w = SimpleWeightedGraph(g2)
weights = rand(1:10, ne(g2))
for e in edges(g2)
    add_edge!(g2_w, src(e), dst(e), weights[e.src])
end

w_test_graphs = [g1_w, g2_w]

@testset "shortest_path vector" begin

    ## Boolean weights
    for g in test_graphs
        n = nv(g)
        # Convert to GPU format
        A_int = convert(SparseMatrixCSC{Int32,Int32}, adjacency_matrix(g, Bool; dir = :out))
        A_bool = convert(SparseMatrixCSC{Bool,Int32}, A_int)


        A_T_gpu_int = SparseGPUMatrixSELL(transpose(A_int), TEST_BACKEND)
        A_T_gpu_bool = SparseGPUMatrixSELL(transpose(A_bool), TEST_BACKEND)

        # get reference result
        ref = gdistances(g, 1)

        # Run BFS
        sp_result_bool = shortest_path(A_T_gpu_bool, one(Int32))
        sp_result_int = shortest_path(A_T_gpu_int, one(Int32))

        # Check if the result is correct
        res_bool = zeros(Int32, n)
        res_int = zeros(Int32, n)
        copyto!(res_bool, sp_result_bool)
        copyto!(res_int, sp_result_int)
        @test res_bool == ref
        @test res_int == ref

    end

    ## Integer weights
    for g in w_test_graphs
        n = nv(g)
        # Convert to GPU format
        A = convert(SparseMatrixCSC{Int32,Int32}, adjacency_matrix(g, Int32; dir = :out))

        A_T_gpu = SparseGPUMatrixSELL(transpose(A), TEST_BACKEND)

        # get reference result
        ref = dijkstra_shortest_paths(g, 1).dists
        # Run shortest path
        sp_result = shortest_path(A_T_gpu, one(Int32))
        # Check if the result is correct
        res = zeros(Int32, n)
        copyto!(res, sp_result)
        @test res == ref

    end

end

@testset "shortest_path multiple sources" begin
    ## Boolean weights
    for g in test_graphs
        n = nv(g)
        # Convert to GPU format
        A_int = convert(SparseMatrixCSC{Int32,Int32}, adjacency_matrix(g, Bool; dir = :out))
        A_bool = convert(SparseMatrixCSC{Bool,Int32}, A_int)
        A_T_gpu_int = SparseGPUMatrixSELL(transpose(A_int), TEST_BACKEND)
        A_T_gpu_bool = SparseGPUMatrixSELL(transpose(A_bool), TEST_BACKEND)
        sources = convert(Vector{Int32}, collect(1:n))
        # get reference result
        ref = zeros(Int32, n, n)
        for i = 1:n
            ref[:, i] = gdistances(g, i)
        end
        res_int = shortest_path(A_T_gpu_int, sources)
        res_bool = shortest_path(A_T_gpu_bool, sources)
        # Check if the result is correct
        res_bool_cpu = zeros(Int32, n, n)
        res_int_cpu = zeros(Int32, n, n)
        copyto!(res_bool_cpu, res_bool)
        copyto!(res_int_cpu, res_int)
        @test res_bool_cpu == ref
        @test res_int_cpu == ref
    end

    ## Integer weights
    for g in w_test_graphs
        n = nv(g)
        # Convert to GPU format
        A = convert(SparseMatrixCSC{Int32,Int32}, adjacency_matrix(g, Int32; dir = :out))
        A_T_gpu = SparseGPUMatrixSELL(transpose(A), TEST_BACKEND)
        sources = convert(Vector{Int32}, collect(1:n))
        # get reference result
        ref = zeros(Int32, n, n)
        for i = 1:n
            ref[:, i] = dijkstra_shortest_paths(g, i).dists
        end
        res = shortest_path(A_T_gpu, sources)
        # Check if the result is correct
        res_cpu = zeros(Int32, n, n)
        copyto!(res_cpu, res)
        @test res_cpu == ref
    end
end
