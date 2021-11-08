#: Concrete type

export LinearOperator, linear_operator
export IdentityOperator, identity_operator
export domain_size, range_size


# Generic linear operator

struct LinearOperator<:AbstractLinearOperator
    domain_size::NTuple{<:Any,Int64}
    range_size::NTuple{<:Any,Int64}
    matvecprod::Function
    matvecprod_adj::Function
end

domain_size(L::LinearOperator) = L.domain_size
range_size(L::LinearOperator) = L.range_size
matvecprod(L::LinearOperator, u::AbstractArray) = L.matvecprod(u)
matvecprod_adj(L::LinearOperator, v::AbstractArray) = L.matvecprod_adj(v)

linear_operator(domain_size::NTuple{N1,Int64}, range_size::NTuple{N2,Int64}, matvecprod::Function, matvecprod_adj::Function) where {N1,N2} = LinearOperator(domain_size, range_size, matvecprod, matvecprod_adj)


# Identity

struct IdentityOperator<:AbstractLinearOperator
    size::NTuple{<:Any,Int64}
end

domain_size(I::IdentityOperator) = I.size
range_size(I::IdentityOperator) = I.size
matvecprod(::IdentityOperator, u::AbstractArray) = u
matvecprod_adj(::IdentityOperator, v::AbstractArray) = v

identity_operator(DT::DataType, size::NTuple{N,Int64}) where N = IdentityOperator(size)