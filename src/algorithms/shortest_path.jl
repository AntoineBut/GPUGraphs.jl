########
# Shortest Path Algorithm
# Based on the Bellman-Ford algorithm, implemented with a masked sparse matrix-vector multiplication
# Supports both unweighted (Boolean weights) and weighted (integer or float weights) graphs
# For unweighted graphs, the distance type is Int32
# For weighted graphs, the distance type is the same as the weight type
########


### Single source shortest path

function shortest_path(
    A_T::TM,
    source::Ti;
) where {Tv,Ti<:Integer,TM<:AbstractSparseGPUMatrix{Tv,Ti}}
    backend = get_backend(A_T)
    # Tv is the type of values in the (weighted) incidence matrix, typically Float32
    T_dist = ifelse(Tv == Bool, Int32, Tv)
    # T_dist is the type of distances, which is kept identical to Tv except for Boolean weights where we use Int32 distances
    next = KernelAbstractions.zeros(backend, T_dist, size(A_T, 1))
    updated = KernelAbstractions.zeros(backend, Ti, size(A_T, 1))
    diff = KernelAbstractions.zeros(backend, Ti, size(A_T, 1))

    # Ti is typically Int32, and is guaranteed to be able to hold the size of the matrix
    dist = KernelAbstractions.allocate(backend, T_dist, size(A_T, 1))

    dist .= typemax(T_dist)
    next .= typemax(T_dist)

    shortest_path!(A_T, source, dist, next, updated, diff)
    return dist
end

function shortest_path!(
    A_T::TM,
    source::Ti,
    dist::TVd,
    next::TVd,
    updated::TVi,
    diff::TVi,
) where {
    Tv,
    Td,
    Ti<:Integer,
    TVd<:AbstractVector{Td},
    TVi<:AbstractVector{Ti},
    TM<:AbstractSparseGPUMatrix{Tv,Ti},
}
    # Every iteration, we mark "updated" on indices changed by the current iteration, 
    # and only iterate from these vertices next iteration
    @allowscalar dist[source] = zero(Td)
    @allowscalar next[source] = zero(Td)
    @allowscalar updated[source] = one(Ti)

    iter = zero(Ti)

    while true

        iter += one(Ti)
        gpu_spmv!(
            next,
            A_T,
            dist,
            mul = GPUGraphs_safe_add,
            add = GPUGraphs_min,
            accum = GPUGraphs_min,
            #mask = updated, Not used yet
        )
        # Diff : where we made progress
        @. diff = next < dist
        if reduce(|, diff) == zero(Td)
            return nothing
        end
        # Update the dist array
        dist .= next

    end
    return nothing
end


### Multiple sources shortest path

function shortest_path(
    A_T::TM,
    sources::TV;
) where {Tv,Ti<:Integer,TV<:AbstractVector{Ti},TM<:AbstractSparseGPUMatrix{Tv,Ti}}
    backend = get_backend(A_T)

    # Tv is the type of values in the (weighted) incidence matrix, typically Float32
    T_dist = ifelse(Tv == Bool, Int32, Tv)
    # T_dist is the type of distances, which is kept identical to Tv except for Boolean weights where we use Int32 distances
    next = KernelAbstractions.zeros(backend, T_dist, size(A_T, 1), size(sources, 1))
    updated = KernelAbstractions.zeros(backend, Ti, size(A_T, 1), size(sources, 1))
    diff = KernelAbstractions.zeros(backend, Ti, size(A_T, 1), size(sources, 1))

    # Ti is typically Int32, and is guaranteed to be able to hold the size of the matrix
    dist = KernelAbstractions.allocate(backend, T_dist, size(A_T, 1), size(sources, 1))

    dist .= typemax(T_dist)
    next .= typemax(T_dist)

    shortest_path!(A_T, sources, dist, next, updated, diff)

    return dist
end

function shortest_path!(
    A_T::TM,
    sources::TVs,
    dist::TMd,
    next::TMd,
    updated::TMi,
    diff::TMi,
) where {
    Tv,
    Td,
    Ti<:Integer,
    TVs<:AbstractVector{Ti},
    TMd<:AbstractMatrix{Td},
    TMi<:AbstractMatrix{Ti},
    TM<:AbstractSparseGPUMatrix{Tv,Ti},
}
    # Every iteration, we mark "updated" on indices changed by the current iteration, 
    # and only iterate from these vertices next iteration
    for i in eachindex(sources)
        s = sources[i]
        @allowscalar dist[s, i] = zero(Td)
        @allowscalar next[s, i] = zero(Td)
        @allowscalar updated[s, i] = one(Ti)
    end


    iter = zero(Ti)

    while true
        iter += one(Ti)
        gpu_spmm!(
            next,
            A_T,
            dist,
            mul = GPUGraphs_safe_add,
            add = GPUGraphs_min,
            accum = GPUGraphs_min,
            #mask = updated, Not used yet
        )
        # Diff : where we made progress
        @. diff = next < dist
        if reduce(|, diff) == zero(Td)
            return nothing
        end
        # Update the dist array
        dist .= next

    end
    return nothing
end
