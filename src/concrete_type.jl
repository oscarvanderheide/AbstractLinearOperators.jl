#: Concrete type

export LinearOperator, linear_operator


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