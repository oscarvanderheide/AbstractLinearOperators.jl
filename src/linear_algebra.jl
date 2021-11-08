#: Linear algebra for AbstractLinearOperator


# Auxiliary types

## ScaledLinearOperators: c*A

struct ScaledLinearOperator<:AbstractLinearOperator
    c::Number
    A::AbstractLinearOperator
end

domain_size(A::ScaledLinearOperator) = domain_size(A.A)
range_size(A::ScaledLinearOperator) = range_size(A.A)
matvecprod(A::ScaledLinearOperator, u::AbstractArray) = A.c*(A.A*u)
matvecprod_adj(A::ScaledLinearOperator, v::AbstractArray) = conj(A.c)*matvecprod_adj(A.A, v)

*(c::Number, A::AbstractLinearOperator) = ScaledLinearOperator(c, A)

## PlusLinearOperators: A+B

struct PlusLinearOperator<:AbstractLinearOperator
    A::AbstractLinearOperator
    B::AbstractLinearOperator
end

domain_size(A::PlusLinearOperator) = domain_size(A.A)
range_size(A::PlusLinearOperator) = range_size(A.A)
matvecprod(A::PlusLinearOperator, u::AbstractArray) = matvecprod(A.A, u)+matvecprod(A.B, u)
matvecprod_adj(A::PlusLinearOperator, v::AbstractArray) = matvecprod_adj(A.A, v)+matvecprod_adj(A.B, v)

+(A::AbstractLinearOperator, B::AbstractLinearOperator) = PlusLinearOperator(A, B)

## MinusLinearOperators: A-B

struct MinusLinearOperator<:AbstractLinearOperator
    A::AbstractLinearOperator
    B::AbstractLinearOperator
end

domain_size(A::MinusLinearOperator) = domain_size(A.A)
range_size(A::MinusLinearOperator) = range_size(A.A)
matvecprod(A::MinusLinearOperator, u::AbstractArray) = matvecprod(A.A, u)-matvecprod(A.B, u)
matvecprod_adj(A::MinusLinearOperator, v::AbstractArray) = matvecprod_adj(A.A, v)-matvecprod_adj(A.B, v)

-(A::AbstractLinearOperator, B::AbstractLinearOperator) = MinusLinearOperator(A, B)

## MultLinearOperators: A*B

struct MultLinearOperator<:AbstractLinearOperator
    A::AbstractLinearOperator
    B::AbstractLinearOperator
end

domain_size(A::MultLinearOperator) = domain_size(A.B)
range_size(A::MultLinearOperator) = range_size(A.A)
matvecprod(A::MultLinearOperator, u::AbstractArray) = matvecprod(A.A, matvecprod(A.B, u))
matvecprod_adj(A::MultLinearOperator, v::AbstractArray) = matvecprod_adj(A.B, matvecprod_adj(A.A, v))

*(A::AbstractLinearOperator, B::AbstractLinearOperator) = MultLinearOperator(A, B)

## AdjointLinearOperators: adjoint(A)

struct AdjointLinearOperator<:AbstractLinearOperator
    A::AbstractLinearOperator
end

domain_size(A::AdjointLinearOperator) = range_size(A.A)
range_size(A::AdjointLinearOperator) = domain_size(A.A)
matvecprod(A::AdjointLinearOperator, u::AbstractArray) = matvecprod_adj(A.A, u)
matvecprod_adj(A::AdjointLinearOperator, v::AbstractArray) = matvecprod(A.A, v)

adjoint(A::AbstractLinearOperator) = AdjointLinearOperator(A)