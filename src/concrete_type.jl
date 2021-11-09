#: Concrete type

export LinearOperator, linear_operator
export IdentityOperator, identity_operator
export domain_size, range_size


# Generic linear operator

struct LinearOperator{T,ND,NR}<:AbstractLinearOperator{T,ND,NR}
    domain_size::NTuple{<:Any,Int64}
    range_size::NTuple{<:Any,Int64}
    matvecprod::Function
    matvecprod_adj::Function
end

domain_size(L::LinearOperator) = L.domain_size
range_size(L::LinearOperator) = L.range_size
matvecprod(L::LinearOperator{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = L.matvecprod(u)
matvecprod_adj(L::LinearOperator{T,ND,NR}, v::AbstractArray{T,NR}) where {T,ND,NR} = L.matvecprod_adj(v)

linear_operator(T::DataType, domain_size::NTuple{ND,Int64}, range_size::NTuple{NR,Int64}, matvecprod::Function, matvecprod_adj::Function) where {ND,NR} = LinearOperator{T,ND,NR}(domain_size, range_size, matvecprod, matvecprod_adj)


# Identity

struct IdentityOperator{T,N}<:AbstractLinearOperator{T,N,N}
    size::NTuple{<:Any,Int64}
end

domain_size(I::IdentityOperator) = I.size
range_size(I::IdentityOperator) = I.size
matvecprod(::IdentityOperator{T,N}, u::AbstractArray{T,N}) where {T,N} = u
matvecprod_adj(::IdentityOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = v

identity_operator(T::DataType, size::NTuple{N,Int64}) where N = IdentityOperator{T,N}(size)