# Implementation of neutral element for operations in Base

"""
    monoid_neutral(T, op)

Return the neutral element of the operation `op` for the type `T`.
"""
monoid_neutral(::Type{T}, ::typeof(+)) where {T} = zero(T)
monoid_neutral(::Type{T}, ::typeof(*)) where {T} = one(T)
monoid_neutral(::Type{T}, ::typeof(min)) where {T} = typemax(T)
monoid_neutral(::Type{T}, ::typeof(max)) where {T} = typemin(T)
monoid_neutral(::Type{T}, ::typeof(&)) where {T} = ~zero(T)
monoid_neutral(::Type{T}, ::typeof(|)) where {T} = zero(T) # TODO : Is this always true?
monoid_neutral(::Type{T}, ::typeof(^)) where {T} = zero(T) # TODO : Is this always true?
monoid_neutral(::Type{T}, ::typeof(⊽)) where {T} = ~zero(T) # TODO : Is this always true?




# Implementation of commomly used binary operators
# These function all have the same inputs : (x, y, A_i, A_j, B_i, B_j)

GPUGraphs_first(x, _, _, _, _, _) = x

GPUGraphs_second(_, y, _, _, _, _) = y
GPUGraphs_any(_, _, _, _, _, _) = 1
# TODO Assumes the function is only called on stored values of A (true so far in spmv kernel, might not always be true)
GPUGraphs_pair(_, y, _, _, _, _) = ifelse(y != zero(y), 1, 0)
GPUGraphs_add(x, y, _, _, _, _) = x + y
monoid_neutral(::Type{T}, ::typeof(GPUGraphs_add)) where {T} = zero(T)

GPUGraphs_minus(x, y, _, _, _, _) = x - y
GPUGraphs_rminus(x, y, _, _, _, _) = y - x
GPUGraphs_mul(x, y, _, _, _, _) = x * y
monoid_neutral(::Type{T}, ::typeof(GPUGraphs_mul)) where {T} = one(T)

GPUGraphs_div(x, y, _, _, _, _) = x / y
GPUGraphs_rdiv(x, y, _, _, _, _) = x \ y
GPUGraphs_pow(x, y, _, _, _, _) = x^y
GPUGraphs_iseq(x::T, y::T, _, _, _, _) where {T} = T(x == y)
GPUGraphs_isne(x::T, y::T, _, _, _, _) where {T} = T(x != y)
GPUGraphs_min(x, y, _, _, _, _) = min(x, y)
GPUGraphs_max(x, y, _, _, _, _) = max(x, y)
GPUGraphs_isgt(x::T, y::T, _, _, _, _) where {T} = T(x > y)
GPUGraphs_islt(x::T, y::T, _, _, _, _) where {T} = T(x < y)
GPUGraphs_isge(x::T, y::T, _, _, _, _) where {T} = T(x >= y)
GPUGraphs_isle(x::T, y::T, _, _, _, _) where {T} = T(x <= y)

GPUGraphs_lor(x, y, _, _, _, _) = x | y
monoid_neutral(::Type{Bool}, ::typeof(GPUGraphs_lor)) = false
GPUGraphs_land(x, y, _, _, _, _) = x & y
monoid_neutral(::Type{Bool}, ::typeof(GPUGraphs_land)) = true
# LXOR ?

GPUGraphs_eq(x, y, _, _, _, _) = x == y
GPUGraphs_ne(x, y, _, _, _, _) = x != y
GPUGraphs_gt(x, y, _, _, _, _) = x > y
GPUGraphs_lt(x, y, _, _, _, _) = x < y
GPUGraphs_ge(x, y, _, _, _, _) = x >= y
GPUGraphs_le(x, y, _, _, _, _) = x <= y

##
##

GPUGraphs_bor(x, y, _, _, _, _) = x | y
monoid_neutral(::Type{T}, ::typeof(GPUGraphs_bor)) where {T} = zero(T)
GPUGraphs_band(x, y, _, _, _, _) = x & y
monoid_neutral(::Type{T}, ::typeof(GPUGraphs_band)) where {T} = ~zero(T)
GPUGraphs_bxor(x, y, _, _, _, _) = x ⊻ y
GPUGraphs_bshiftr(x, y, _, _, _, _) = x << y
GPUGraphs_bshiftl(x, y, _, _, _, _) = x >> y

GPUGraphs_firsti(_, _, A_i, _, _, _) = A_i
GPUGraphs_firstj(_, _, _, A_j, _, _) = A_j

GPUGraphs_secondi(_, _, _, _, B_i, _) = B_i
GPUGraphs_secondj(_, _, _, _, _, B_j) = B_j
