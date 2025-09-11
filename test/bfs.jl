TEST_BACKEND = if get(ENV, "CI", "false") == "false"
    
    #Metal.MetalBackend()  # our personal laptops
    CUDA.CUDABackend()  # on the cluster
# KernelAbstractions.CPU()
else
    KernelAbstractions.CPU()
end

@testset "bfs" begin
    g1 = barabasi_albert(30, 3)
    g2 = dorogovtsev_mendes(30)
    #g3 = grid((10, 10))

    test_graphs = [g1, g2]

    for g in test_graphs
        n = nv(g)
        # Convert to GPU format
        A_int = convert(SparseMatrixCSC{Int32,Int32}, adjacency_matrix(g, Bool; dir = :out))
        A_bool = convert(SparseMatrixCSC{Bool,Int32}, A_int)


        A_T_gpu_int = SparseGPUMatrixCSR(transpose(A_int), TEST_BACKEND)
        A_T_gpu_bool = SparseGPUMatrixCSR(transpose(A_bool), TEST_BACKEND)
        println(A_T_gpu_int)
        println(A_T_gpu_bool)

        # get reference result
        ref = gdistances(g, 1)

        # Run BFS
        bfs_result_bool = bfs_distances(A_T_gpu_bool, one(Int32))
        bfs_result_int = bfs_distances(A_T_gpu_int, one(Int32))

        # Check if the result is correct
        res_bool = zeros(Int32, n)
        res_int = zeros(Int32, n)
        copyto!(res_bool, bfs_result_bool)
        copyto!(res_int, bfs_result_int)
        @test res_bool == ref
        @test res_int == ref

    end

end
