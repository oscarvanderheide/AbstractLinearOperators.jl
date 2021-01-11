#: Linear algebra for AbstractLinearOperator


# Auxiliary types

## ScaledLinearOperators: c*A

struct ScaledLinearOperator{DT,RT} <: AbstractLinearOperator{DT,RT}
    c::Number
    A::AbstractLinearOperator{DT,RT}
end

domain_size(A::ScaledLinearOperator) = domain_size(A.A)
range_size(A::ScaledLinearOperator) = range_size(A.A)
matvecprod(A::ScaledLinearOperator{DT,RT}, u::DT) where {DT,RT} = A.c*matvecprod(A.A, u)
matvecprod_adj(A::ScaledLinearOperator{DT,RT}, v::RT) where {DT,RT} = conj(A.c)*matvecprod_adj(A.A, v)

## PlusLinearOperators: A+B

struct PlusLinearOperator{DT,RT} <: AbstractLinearOperator{DT,RT}
    A::AbstractLinearOperator{DT,RT}
    B::AbstractLinearOperator{DT,RT}
end

domain_size(A::PlusLinearOperator) = domain_size(A.A)
range_size(A::PlusLinearOperator) = range_size(A.A)
matvecprod(A::PlusLinearOperator{DT,RT}, u::DT) where {DT,RT} = matvecprod(A.A, u)+matvecprod(A.B, u)
matvecprod_adj(A::PlusLinearOperator{DT,RT}, v::RT) where {DT,RT} = matvecprod_adj(A.A, v)+matvecprod_adj(A.B, v)

## MinusLinearOperators: A-B

struct MinusLinearOperator{DT,RT} <: AbstractLinearOperator{DT,RT}
    A::AbstractLinearOperator{DT,RT}
    B::AbstractLinearOperator{DT,RT}
end

domain_size(A::MinusLinearOperator) = domain_size(A.A)
range_size(A::MinusLinearOperator) = range_size(A.A)
matvecprod(A::MinusLinearOperator{DT,RT}, u::DT) where {DT,RT} = matvecprod(A.A, u)-matvecprod(A.B, u)
matvecprod_adj(A::MinusLinearOperator{DT,RT}, v::RT) where {DT,RT} = matvecprod_adj(A.A, v)-matvecprod_adj(A.B, v)

## MultLinearOperators: A*B

struct MultLinearOperator{DTB,RTB,RTA} <: AbstractLinearOperator{DTB,RTA}
    A::AbstractLinearOperator{RTB,RTA}
    B::AbstractLinearOperator{DTB,RTB}
end

domain_size(A::MultLinearOperator) = domain_size(A.B)
range_size(A::MultLinearOperator) = range_size(A.A)
matvecprod(A::MultLinearOperator{DTB,RTB,RTA}, u::DTB) where {DTB,RTB,RTA} = matvecprod(A.A, matvecprod(A.B, u))
matvecprod_adj(A::MultLinearOperator{DTB,RTB,RTA}, v::RTA) where {DTB,RTB,RTA} = matvecprod_adj(A.B, matvecprod_adj(A.A, v))

## AdjointLinearOperators: adjoint(A)

struct AdjointLinearOperator{DT,RT} <: AbstractLinearOperator{DT,RT}
    A::AbstractLinearOperator{RT,DT}
end

domain_size(A::AdjointLinearOperator) = range_size(A.A)
range_size(A::AdjointLinearOperator) = domain_size(A.A)
matvecprod(A::AdjointLinearOperator{DT,RT}, u::DT) where {DT,RT} = matvecprod_adj(A.A, u)
matvecprod_adj(A::AdjointLinearOperator{DT,RT}, v::RT) where {DT,RT} = matvecprod(A.A, v)


# Algebra

*(A::AbstractLinearOperator{DT,RT}, u::DT) where{DT,RT} = matvecprod(A, u)
adjoint(A::AbstractLinearOperator{DT,RT}) where {DT,RT} = AdjointLinearOperator{RT,DT}(A)
*(c::T, A::AbstractLinearOperator{DT,RT}) where {T,DT,RT} = ScaledLinearOperator{DT,RT}(c, A)
+(A::LO, B::LO) where {DT,RT,LO<:AbstractLinearOperator{DT,RT}} = PlusLinearOperator{DT,RT}(A, B)
-(A::LO, B::LO) where {DT,RT,LO<:AbstractLinearOperator{DT,RT}} = MinusLinearOperator{DT,RT}(A, B)
*(A::LO2, B::LO1) where {DT1,RT1,RT2,LO1<:AbstractLinearOperator{DT1,RT1},LO2<:AbstractLinearOperator{RT1,RT2}} = MultLinearOperator{DT1,RT1,RT2}(A, B)