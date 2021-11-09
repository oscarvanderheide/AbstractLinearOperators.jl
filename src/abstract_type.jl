#: Abstract types
export AbstractLinearOperator, size, size_vec


"""
Expected behavior:
- domain_size(A::AbstractLinearOperator)
- range_size(A::AbstractLinearOperator)
- matvecprod(A::AbstractLinearOperator, u)
- matvecprod_adj(A::AbstractLinearOperator, v)
"""
abstract type AbstractLinearOperator{T,ND,NR} end


# Base functions

size(A::AbstractLinearOperator) = (range_size(A), domain_size(A))
show(::IO, A::AbstractLinearOperator) = info(A)
show(::IO, mime::MIME"text/plain", A::AbstractLinearOperator) = info(A)


# Algebra

*(A::AbstractLinearOperator{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = matvecprod(A, u)


# Utils

size_vec(A::AbstractLinearOperator) = (prod(range_size(A)), prod(domain_size(A)))
info(A::AbstractLinearOperator{T,ND,NR}) where {T,ND,NR} = print("Linear operator, domain ≅ ", T, "^", domain_size(A), ", range ≅ ", T, "^", range_size(A))