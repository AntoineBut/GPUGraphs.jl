
function bfs(A_T::TM, source::Ti) where {Tv,Ti<:Integer,TM<:AbstractSparseGPUMatrix{Tv,Ti}}
    backend = get_backend(A_T)

    curr = KernelAbstractions.zeros(backend, Tv, size(A_T, 1))
    next = KernelAbstractions.zeros(backend, Tv, size(A_T, 1))
    explored = KernelAbstractions.zeros(backend, Bool, size(A_T, 1))
    dist = KernelAbstractions.allocate(backend, Tv, size(A_T, 1))

    dist .= typemax(Tv)

    bfs!(A_T, source, dist, curr, next, explored)

    return dist
end

function bfs!(
    A_T::TM,
    source::Ti,
    dist::TV,
    curr::TV,
    next::TV,
    explored::TVb,
) where {
    Tv,
    Ti<:Integer,
    TVb<:AbstractVector{Bool},
    TV<:AbstractVector{Tv},
    TM<:AbstractSparseGPUMatrix{Tv,Ti},
}

    @allowscalar curr[source] = one(Tv)
    @allowscalar dist[source] = zero(Tv)
    @allowscalar explored[source] = true
    iter = zero(Tv)
    while true
        iter += one(Tv)
        next .= zero(Tv)

        gpu_spmv!(next, A_T, curr, &, |)
        # set curr to next on newly explored nodes
        curr .= next .& .!(explored)
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
