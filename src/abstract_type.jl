#: Abstract types

export AbstractLinearOperator


"""
Expected behavior:
- domain_size(A::AbstractLinearOperator)
- range_size(A::AbstractLinearOperator)
- matvecprod(A::AbstractLinearOperator{DT,RT}, u::DT)::RT
- matvecprod_adj(A::AbstractLinearOperator{DT,RT}, v::RT)::DT
"""
RealOrComplex{T} = Union{T,Complex{T}}
abstract type AbstractLinearOperator{T<:AbstractFloat,DT<:AbstractArray{<:RealOrComplex{T}},RT<:AbstractArray{<:RealOrComplex{T}}} end


# Base functions

size(A::AbstractLinearOperator) = (range_size(A), domain_size(A))
show(io::IO, A::AbstractLinearOperator) = info(A)
show(io::IO, mime::MIME"text/plain", A::AbstractLinearOperator) = info(A)