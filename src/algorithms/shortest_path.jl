function shortest_path(
    A_T::TM,
    source::Ti;
) where {Tv,Ti<:Integer,TM<:AbstractSparseGPUMatrix{Tv,Ti}}
    backend = get_backend(A_T)

    # Tv is the type of values in the (weighted) incidence matrix, typically Float32
    next = KernelAbstractions.zeros(backend, Tv, size(A_T, 1))
    updated = KernelAbstractions.zeros(backend, Ti, size(A_T, 1))
    diff = KernelAbstractions.zeros(backend, Ti, size(A_T, 1))

    # Ti is typically Int32, and is guaranteed to be able to hold the size of the matrix
    dist = KernelAbstractions.allocate(backend, Tv, size(A_T, 1))

    dist .= typemax(Tv)
    next .= typemax(Tv)

    shortest_path!(A_T, source, dist, next, updated, diff)

    return dist
end

function shortest_path!(
    A_T::TM,
    source::Ti,
    dist::TVv,
    next::TVv,
    updated::TVi,
    diff::TVi,
) where {
    Tv,
    Ti<:Integer,
    TVv<:AbstractVector{Tv},
    TVi<:AbstractVector{Ti},
    TM<:AbstractSparseGPUMatrix{Tv,Ti},
}
    # Every iteration, we mark "updated" on indeces changed by the current iteration, 
    # and only iterate from these vertices next iteration
    @allowscalar dist[source] = zero(Tv)
    @allowscalar next[source] = zero(Tv)
    @allowscalar updated[source] = one(Ti)

    iter = zero(Ti)

    while true
        iter += one(Ti)
        gpu_spmv!(
            next,
            A_T,
            dist,
            mul = GPUGraphs_add,
            add = GPUGraphs_min,
            accum = GPUGraphs_min,
            #mask = updated, Not used yet
        )
        # Diff : where we made progress
        @. diff = next < dist
        if reduce(|, diff) == zero(Tv)
            return nothing
        end
        # Update the dist array
        dist .= next

    end
    return nothing
end
