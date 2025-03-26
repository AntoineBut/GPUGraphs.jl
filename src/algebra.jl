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
