#: Linear algebra for AbstractLinearOperator


# Auxiliary types

## ScaledLinearOperators: c*A

struct ScaledLinearOperator{T,ND,NR}<:AbstractLinearOperator{T,ND,NR}
    c::T
    A::AbstractLinearOperator{T,ND,NR}
end

domain_size(A::ScaledLinearOperator) = domain_size(A.A)
range_size(A::ScaledLinearOperator) = range_size(A.A)
matvecprod(A::ScaledLinearOperator{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = A.c*(A.A*u)
matvecprod_adj(A::ScaledLinearOperator{T,ND,NR}, v::AbstractArray{T,NR}) where {T,ND,NR} = conj(A.c)*matvecprod_adj(A.A, v)

*(c::T, A::AbstractLinearOperator{T,ND,NR}) where {T,ND,NR} = ScaledLinearOperator{T,ND,NR}(c, A)

Flux.gpu(A::ScaledLinearOperator{T,ND,NR}) where {T,ND,NR} = ScaledLinearOperator{T,ND,NR}(A.c, gpu(A.A))
Flux.cpu(A::ScaledLinearOperator{T,ND,NR}) where {T,ND,NR} = ScaledLinearOperator{T,ND,NR}(A.c, cpu(A.A))

## PlusLinearOperators: A+B

struct PlusLinearOperator{T,ND,NR}<:AbstractLinearOperator{T,ND,NR}
    A::AbstractLinearOperator{T,ND,NR}
    B::AbstractLinearOperator{T,ND,NR}
end

domain_size(A::PlusLinearOperator) = domain_size(A.A)
range_size(A::PlusLinearOperator) = range_size(A.A)
matvecprod(A::PlusLinearOperator{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = matvecprod(A.A, u)+matvecprod(A.B, u)
matvecprod_adj(A::PlusLinearOperator{T,ND,NR}, v::AbstractArray{T,NR}) where {T,ND,NR} = matvecprod_adj(A.A, v)+matvecprod_adj(A.B, v)

+(A::AbstractLinearOperator{T,ND,NR}, B::AbstractLinearOperator{T,ND,NR}) where {T,ND,NR} = PlusLinearOperator{T,ND,NR}(A, B)

Flux.gpu(A::PlusLinearOperator{T,ND,NR}) where {T,ND,NR} = PlusLinearOperator{T,ND,NR}(gpu(A.A), gpu(A.B))
Flux.cpu(A::PlusLinearOperator{T,ND,NR}) where {T,ND,NR} = PlusLinearOperator{T,ND,NR}(cpu(A.A), cpu(A.B))

## MinusLinearOperators: A-B

struct MinusLinearOperator{T,ND,NR}<:AbstractLinearOperator{T,ND,NR}
    A::AbstractLinearOperator{T,ND,NR}
    B::AbstractLinearOperator{T,ND,NR}
end

domain_size(A::MinusLinearOperator) = domain_size(A.A)
range_size(A::MinusLinearOperator) = range_size(A.A)
matvecprod(A::MinusLinearOperator{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = matvecprod(A.A, u)-matvecprod(A.B, u)
matvecprod_adj(A::MinusLinearOperator{T,ND,NR}, v::AbstractArray{T,NR}) where {T,ND,NR} = matvecprod_adj(A.A, v)-matvecprod_adj(A.B, v)

-(A::AbstractLinearOperator{T,ND,NR}, B::AbstractLinearOperator{T,ND,NR}) where {T,ND,NR} = MinusLinearOperator{T,ND,NR}(A, B)

Flux.gpu(A::MinusLinearOperator{T,ND,NR}) where {T,ND,NR} = MinusLinearOperator{T,ND,NR}(gpu(A.A), gpu(A.B))
Flux.cpu(A::MinusLinearOperator{T,ND,NR}) where {T,ND,NR} = MinusLinearOperator{T,ND,NR}(cpu(A.A), cpu(A.B))

## MultLinearOperators: A*B

struct MultLinearOperator{T,ND,Nh,NR}<:AbstractLinearOperator{T,ND,NR}
    A::AbstractLinearOperator{T,Nh,NR}
    B::AbstractLinearOperator{T,ND,Nh}
end

domain_size(A::MultLinearOperator) = domain_size(A.B)
range_size(A::MultLinearOperator) = range_size(A.A)
matvecprod(A::MultLinearOperator{T,ND,Nh,NR}, u::AbstractArray{T,ND}) where {T,ND,Nh,NR} = matvecprod(A.A, matvecprod(A.B, u))
matvecprod_adj(A::MultLinearOperator{T,ND,Nh,NR}, v::AbstractArray{T,NR}) where {T,ND,Nh,NR} = matvecprod_adj(A.B, matvecprod_adj(A.A, v))

*(A::AbstractLinearOperator{T,Nh,NR}, B::AbstractLinearOperator{T,ND,Nh}) where {T,ND,Nh,NR} = MultLinearOperator{T,ND,Nh,NR}(A, B)

Flux.gpu(A::MultLinearOperator{T,ND,Nh,NR}) where {T,ND,Nh,NR} = MultLinearOperator{T,ND,Nh,NR}(gpu(A.A), gpu(A.B))
Flux.cpu(A::MultLinearOperator{T,ND,Nh,NR}) where {T,ND,Nh,NR} = MultLinearOperator{T,ND,Nh,NR}(cpu(A.A), cpu(A.B))

## AdjointLinearOperators: adjoint(A)

struct AdjointLinearOperator{T,ND,NR}<:AbstractLinearOperator{T,ND,NR}
    A::AbstractLinearOperator{T,NR,ND}
end

domain_size(A::AdjointLinearOperator) = range_size(A.A)
range_size(A::AdjointLinearOperator) = domain_size(A.A)
matvecprod(A::AdjointLinearOperator{T,ND,NR}, u::AbstractArray{T,ND}) where {T,ND,NR} = matvecprod_adj(A.A, u)
matvecprod_adj(A::AdjointLinearOperator{T,ND,NR}, v::AbstractArray{T,NR}) where {T,ND,NR} = matvecprod(A.A, v)

adjoint(A::AbstractLinearOperator{T,ND,NR}) where {T,ND,NR} = AdjointLinearOperator{T,NR,ND}(A)

Flux.gpu(A::AdjointLinearOperator{T,ND,NR}) where {T,ND,NR} = AdjointLinearOperator{T,ND,NR}(gpu(A.A))
Flux.cpu(A::AdjointLinearOperator{T,ND,NR}) where {T,ND,NR} = AdjointLinearOperator{T,ND,NR}(cpu(A.A))