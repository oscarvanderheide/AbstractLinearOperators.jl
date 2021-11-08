#: Abstract types
export AbstractLinearOperator, size, domain, range, size_vec, deltype, dtype, reltype, rtype


"""
Expected behavior:
- domain_size(A::AbstractLinearOperator)
- range_size(A::AbstractLinearOperator)
- matvecprod(A::AbstractLinearOperator{DT,RT}, u::DT)::RT
- matvecprod_adj(A::AbstractLinearOperator{DT,RT}, v::RT)::DT
"""
abstract type AbstractLinearOperator{DT<:AbstractArray,RT<:AbstractArray} end


# Base functions

size(A::AbstractLinearOperator) = (range_size(A), domain_size(A))
show(::IO, A::AbstractLinearOperator) = info(A)
show(::IO, mime::MIME"text/plain", A::AbstractLinearOperator) = info(A)


# Utils

domain(::AbstractLinearOperator{DT,RT}) where {DT,RT} = DT
range(::AbstractLinearOperator{DT,RT}) where {DT,RT} = RT
size_vec(A::AbstractLinearOperator) = (prod(range_size(A)), prod(domain_size(A)))
deltype(::AbstractLinearOperator{DT,RT}) where {DT,RT} = eltype(DT)
dtype(::AbstractLinearOperator{DT,RT}) where {DT,RT} = DT
reltype(::AbstractLinearOperator{DT,RT}) where {DT,RT} = eltype(RT)
rtype(::AbstractLinearOperator{DT,RT}) where {DT,RT} = RT
info(A::AbstractLinearOperator) = print(
    "Linear operator, domain: size(", domain(A), ") = ", domain_size(A), "\n",
    "                 range:  size(", range(A), ") = ", range_size(A))