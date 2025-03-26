# Simple structs and function for algebraic semirings and monoids


"""

Monoid{T, F}
A monoid is a set equipped with a binary operation that is associative and has an identity element.


struct Monoid{F<:Function}
    op::F
end

	Semiring{A, B, P, Q}
A semiring is a combinaison of 2 sets and two binary operations. 
The first operation is an arbitrary binary operation from the first set to the second set.
The second is a monoid over the second set.

struct Semiring{P<:Function,Q<:Function}
    add::P
    mul::Q
    
    Semiring(mul::Q, add::P, zero::B, one::A) where {A<:Real,B<:Real,P<:Function,Q<:Function} = new{A, B, P, Q}(mul, add)
end
"""
