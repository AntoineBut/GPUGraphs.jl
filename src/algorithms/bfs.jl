function bfs_distances(
    A_T::TM,
    source::Ti;
    use_mask=true,
) where {Tv,Ti<:Integer,TM<:AbstractSparseGPUMatrix{Tv,Ti}}
    backend = get_backend(A_T)

    # Tv is typically Bool
    curr = KernelAbstractions.zeros(backend, Tv, size(A_T, 1))
    next = KernelAbstractions.zeros(backend, Tv, size(A_T, 1))
    to_explore = KernelAbstractions.ones(backend, Tv, size(A_T, 1))


    # Ti is typically Int32, and is guaranteed to be able to hold the size of the matrix
    # and the maximum value of the distance
    dist = KernelAbstractions.allocate(backend, Ti, size(A_T, 1))

    dist .= typemax(Ti)

    if use_mask
        # BFS with mask
        bfs_distances!(A_T, source, dist, curr, next, to_explore)
    else
        # BFS without mask
        no_mask_bfs_distances!(A_T, source, dist, curr, next, to_explore)
    end

    return dist
end

function bfs_distances!(
    A_T::TM,
    source::Ti,
    dist::TVi,
    curr::TVv,
    next::TVv,
    to_explore::TVv
) where {
    Tv,
    Ti<:Integer,
    TVv<:AbstractVector{Tv},
    TVi<:AbstractVector{Ti},
    TM<:AbstractSparseGPUMatrix{Tv,Ti},
}
    @allowscalar curr[source] = one(Tv)
    @allowscalar dist[source] = zero(Ti)
    @allowscalar to_explore[source] = zero(Tv)
    iter = zero(Ti)
    next .= zero(Tv)

    while true
        iter += one(Ti)        
        gpu_spmv!(next, A_T, curr, mul=GPUGraphs_band, add=GPUGraphs_bor, mask=to_explore)
        if reduce(|, next) == zero(Tv)
            return nothing
        end
        # Update the dist array where curr is not zero and dist is zero
        dist .= ifelse.(next .== one(Tv), iter, dist)

        # set to_explore to false for the newly explored nodes
        to_explore .= to_explore .& (next .== zero(Tv))
        # set curr to next
        curr .= next
        next .= zero(Tv)
    end
    return nothing
end

function no_mask_bfs_distances!(
    A_T::TM,
    source::Ti,
    dist::TVi,
    curr::TVv,
    next::TVv,
    to_explore::TVv
) where {
    Tv,
    Ti<:Integer,
    TVv<:AbstractVector{Tv},
    TVi<:AbstractVector{Ti},
    TM<:AbstractSparseGPUMatrix{Tv,Ti},
}
    @allowscalar curr[source] = one(Tv)
    @allowscalar dist[source] = zero(Ti)
    @allowscalar to_explore[source] = zero(Tv)
    iter = zero(Ti)
    next .= zero(Tv)

    while true
        iter += one(Ti)        
        gpu_spmv!(next, A_T, curr, mul=GPUGraphs_band, add=GPUGraphs_bor)
        curr .= next .& to_explore
        if reduce(|, curr) == zero(Tv)
            return nothing
        end
        # Update the dist array where curr is not zero and dist is zero
        dist .= ifelse.(curr .== one(Tv), iter, dist)

        # set to_explore to false for the newly explored nodes
        to_explore .= to_explore .& (curr .== zero(Tv))
        # set curr to next
        next .= zero(Tv)
    end
    return nothing
end