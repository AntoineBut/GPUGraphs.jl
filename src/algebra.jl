# Simple structs and function for algebraic semirings and monoids

"""
	Monoid{T, F}
A monoid is a set equipped with a binary operation that is associative and has an identity element.

"""
struct Monoid{T<:Real,F<:Function}
    op::F
    zero::T
end

"""
	Semiring{A, B, P, Q}
A semiring is a combinaison of 2 sets and two binary operations. 
The first operation is an arbitrary binary operation from the first set to the second set.
The second is a monoid over the second set.
"""
struct Semiring{A<:Real,B<:Real,P<:Function,Q<:Function}
    mul::Q
    add::Monoid{B,P}
    zero::B
    one::A
end
