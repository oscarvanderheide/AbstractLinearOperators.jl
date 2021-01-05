#: Linear algebra for AbstractLinearOperator


# Auxiliary types

## ScaledLinearOperators: c*A

struct ScaledLinearOperator{T,DT,RT} <: AbstractLinearOperator{T,DT,RT}
    c::T
    A::AbstractLinearOperator{T,DT,RT}
end

domain_size(A::ScaledLinearOperator) = domain_size(A.A)
range_size(A::ScaledLinearOperator) = range_size(A.A)
matvecprod(A::ScaledLinearOperator{T,DT,RT}, u::DT) where {T,DT,RT} = A.c*matvecprod(A.A, u)
matvecprod_adj(A::ScaledLinearOperator{T,DT,RT}, v::RT) where {T,DT,RT} = conj(A.c)*matvecprod_adj(A.A, v)

## PlusLinearOperators: A+B

struct PlusLinearOperator{T,DT,RT} <: AbstractLinearOperator{T,DT,RT}
    A::AbstractLinearOperator{T,DT,RT}
    B::AbstractLinearOperator{T,DT,RT}
end

domain_size(A::PlusLinearOperator) = domain_size(A.A)
range_size(A::PlusLinearOperator) = range_size(A.A)
matvecprod(A::PlusLinearOperator{T,DT,RT}, u::DT) where {T,DT,RT} = matvecprod(A.A, u)+matvecprod(A.B, u)
matvecprod_adj(A::PlusLinearOperator{T,DT,RT}, v::RT) where {T,DT,RT} = matvecprod_adj(A.A, v)+matvecprod_adj(A.B, v)

## MinusLinearOperators: A-B

struct MinusLinearOperator{T,DT,RT} <: AbstractLinearOperator{T,DT,RT}
    A::AbstractLinearOperator{T,DT,RT}
    B::AbstractLinearOperator{T,DT,RT}
end

domain_size(A::MinusLinearOperator) = domain_size(A.A)
range_size(A::MinusLinearOperator) = range_size(A.A)
matvecprod(A::MinusLinearOperator{T,DT,RT}, u::DT) where {T,DT,RT} = matvecprod(A.A, u)-matvecprod(A.B, u)
matvecprod_adj(A::MinusLinearOperator{T,DT,RT}, v::RT) where {T,DT,RT} = matvecprod_adj(A.A, v)-matvecprod_adj(A.B, v)

## MultLinearOperators: A*B

struct MultLinearOperator{T,DTB,RTB,RTA} <: AbstractLinearOperator{T,DTB,RTA}
    A::AbstractLinearOperator{T,RTB,RTA}
    B::AbstractLinearOperator{T,DTB,RTB}
end

domain_size(A::MultLinearOperator) = domain_size(A.B)
range_size(A::MultLinearOperator) = range_size(A.A)
matvecprod(A::MultLinearOperator{T,DTB,RTB,RTA}, u::DTB) where {T,DTB,RTB,RTA} = matvecprod(A.A, matvecprod(A.B, u))
matvecprod_adj(A::MultLinearOperator{T,DTB,RTB,RTA}, v::RTA) where {T,DTB,RTB,RTA} = matvecprod_adj(A.B, matvecprod_adj(A.A, v))

## AdjointLinearOperators: adjoint(A)

struct AdjointLinearOperator{T,DT,RT} <: AbstractLinearOperator{T,DT,RT}
    A::AbstractLinearOperator{T,RT,DT}
end

domain_size(A::AdjointLinearOperator) = range_size(A.A)
range_size(A::AdjointLinearOperator) = domain_size(A.A)
matvecprod(A::AdjointLinearOperator{T,DT,RT}, u::DT) where {T,DT,RT} = matvecprod_adj(A.A, u)
matvecprod_adj(A::AdjointLinearOperator{T,DT,RT}, v::RT) where {T,DT,RT} = matvecprod(A.A, v)


# Algebra

*(A::AbstractLinearOperator{T,DT,RT}, u::DT) where{T,DT,RT} = matvecprod(A, u)
adjoint(A::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = AdjointLinearOperator{T,RT,DT}(A)
*(c::T, A::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = ScaledLinearOperator{T,DT,RT}(c, A)
+(A::AbstractLinearOperator{T,DT,RT}, B::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = PlusLinearOperator{T,DT,RT}(A, B)
-(A::AbstractLinearOperator{T,DT,RT}, B::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = MinusLinearOperator{T,DT,RT}(A, B)
*(A::AbstractLinearOperator{T,RT1,RT2}, B::AbstractLinearOperator{T,DT1,RT1}) where {T,DT1,RT1,RT2} = MultLinearOperator{T,DT1,RT1,RT2}(A, B)