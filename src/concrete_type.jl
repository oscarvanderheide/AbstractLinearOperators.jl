#: Concrete type

export LinearOperator, linear_operator
export domain_size, range_size, matvecprod, matvecprod_adj, invmatvecprod, invmatvecprod_adj

struct LinearOperator{T,ND,NR}<:AbstractLinearOperator{T,ND,NR}
    domain_size::NTuple{<:Any,Int64}
    range_size::NTuple{<:Any,Int64}
    matvecprod::Function
    matvecprod_adj::Function
    invmatvecprod::Union{Nothing,Function}
    invmatvecprod_adj::Union{Nothing,Function}
end

domain_size(L::LinearOperator) = L.domain_size
range_size(L::LinearOperator) = L.range_size
matvecprod(L::LinearOperator{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = L.matvecprod(u)
matvecprod_adj(L::LinearOperator{T,ND,NR}, v::AbstractArray{T,NR}) where {T,ND,NR} = L.matvecprod_adj(v)
invmatvecprod(L::LinearOperator{T,ND,NR}, u::AbstractArray{T,NR}) where {T,ND,NR} = ~isnothing(L.invmatvecprod) && L.invmatvecprod(u)
invmatvecprod_adj(L::LinearOperator{T,ND,NR}, v::AbstractArray{T,ND}) where {T,ND,NR} = ~isnothing(L.invmatvecprod_adj) && L.invmatvecprod_adj(v)

linear_operator(T::DataType, domain_size::NTuple{ND,Int64}, range_size::NTuple{NR,Int64}, matvecprod::Function, matvecprod_adj::Function; invmatvecprod::Union{Nothing,Function}=nothing, invmatvecprod_adj::Union{Nothing,Function}=nothing) where {ND,NR} = LinearOperator{T,ND,NR}(domain_size, range_size, matvecprod, matvecprod_adj, invmatvecprod, invmatvecprod_adj)