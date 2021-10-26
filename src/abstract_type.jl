#: Abstract types

export AbstractLinearOperator
export size


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