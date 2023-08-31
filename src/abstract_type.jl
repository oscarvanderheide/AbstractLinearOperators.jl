#: Abstract types
export AbstractLinearOperator

abstract type AbstractLinearOperator{TD<:Number,ND,TR<:Number,NR} end


# Base functions

domain_size(::AbstractLinearOperator) = nothing
range_size(::AbstractLinearOperator) = nothing
label(::AbstractLinearOperator) = nothing
Base.size(A::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = ((TR, range_size(A)...), (TD, domain_size(A)...))
Base.show(::IO, A::AbstractLinearOperator) = info(A)
Base.show(::IO, mime::MIME"text/plain", A::AbstractLinearOperator) = info(A)


# Algebra

Base.:*(A::AbstractLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TD,ND}) where {TD,ND,TR,NR} = matvecprod(A, u)
Base.:\(A::AbstractLinearOperator{TD,ND,TR,NR}, u::AbstractArray{TR,NR}) where {TD,ND,TR,NR} = invmatvecprod(A, u)


# Utils

info(A::AbstractLinearOperator{TD,ND,TR,NR}) where {TD,ND,TR,NR} = print("Linear operator, domain ≅ ", TD, "^", domain_size(A), ", range ≅ ", TR, "^", range_size(A), ~isnothing(label(A)) ? string(", label=", label(A)) : "")