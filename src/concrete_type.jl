#: Concrete type

export LinearOperator, linear_operator
export IdentityOperator, identity_operator


# Generic linear operator

struct LinearOperator{DT,RT}<:AbstractLinearOperator{DT,RT}
    domain_size::NTuple{<:Any,Int64}
    range_size::NTuple{<:Any,Int64}
    matvecprod::Function
    matvecprod_adj::Function
end

domain_size(L::LinearOperator{DT,RT}) where {DT,RT} = L.domain_size
range_size(L::LinearOperator{DT,RT}) where {DT,RT} = L.range_size
matvecprod(L::LinearOperator{DT,RT}, u::DT) where {DT,RT} = L.matvecprod(u)
matvecprod_adj(L::LinearOperator{DT,RT}, v::RT) where {DT,RT} = L.matvecprod_adj(v)

linear_operator(DT::DataType, RT::DataType, domain_size::NTuple{N1,Int64}, range_size::NTuple{N2,Int64}, matvecprod::Function, matvecprod_adj::Function) where {N1,N2} = LinearOperator{DT,RT}(domain_size, range_size, matvecprod, matvecprod_adj)


# Identity

struct IdentityOperator{DT}<:AbstractLinearOperator{DT,DT}
    size::NTuple{<:Any,Int64}
end

domain_size(I::IdentityOperator{DT}) where DT = I.size
range_size(I::IdentityOperator{DT}) where DT = I.size
matvecprod(::IdentityOperator{DT}, u::DT) where DT = u
matvecprod_adj(::IdentityOperator{DT}, v::DT) where DT = v

identity_operator(DT::DataType, size::NTuple{N,Int64}) where N = IdentityOperator{DT}(size)