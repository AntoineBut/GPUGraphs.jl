function bfs_distances(
    A_T::TM,
    source::Ti,
) where {Tv,Ti<:Integer,TM<:AbstractSparseGPUMatrix{Tv,Ti}}
    backend = get_backend(A_T)

    # Tv is typically Bool
    curr = KernelAbstractions.zeros(backend, Tv, size(A_T, 1))
    next = KernelAbstractions.zeros(backend, Tv, size(A_T, 1))
    explored = KernelAbstractions.zeros(backend, Tv, size(A_T, 1))


    # Ti is typically Int32, and is guaranteed to be able to hold the size of the matrix
    # and the maximum value of the distance
    dist = KernelAbstractions.allocate(backend, Ti, size(A_T, 1))

    dist .= typemax(Ti)

    bfs_distances!(A_T, source, dist, curr, next, explored)

    return dist
end

function bfs_distances!(
    A_T::TM,
    source::Ti,
    dist::TVi,
    curr::TVv,
    next::TVv,
    explored::TVv,
) where {
    Tv,
    Ti<:Integer,
    TVv<:AbstractVector{Tv},
    TVi<:AbstractVector{Ti},
    TM<:AbstractSparseGPUMatrix{Tv,Ti},
}
    @allowscalar curr[source] = one(Tv)
    @allowscalar dist[source] = zero(Ti)
    @allowscalar explored[source] = one(Tv)
    iter = zero(Ti)
    #mask = KernelAbstractions.zeros(get_backend(A_T), Ti, size(A_T, 1))
    #copyto!(mask, 1:size(A_T, 1))
    #@allowscalar

    while true
        iter += one(Ti)
        next .= zero(Tv)

        gpu_spmv!(next, A_T, curr, mul=GPUGraphs_band, add=GPUGraphs_bor)
        # set curr to next on newly explored nodes
        curr .= next .& .~(explored)
        if reduce(|, curr) == zero(Tv)
            return nothing
        end
        # Update the dist array where curr is not zero and dist is zero
        dist .= ifelse.(curr .== one(Tv), iter, dist)

        # set explored to true for the newly explored nodes
        explored .= explored .| (curr .== one(Tv))
    end
    return nothing
end
