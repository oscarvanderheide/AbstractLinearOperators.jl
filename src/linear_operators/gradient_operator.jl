export GradientOperator, gradient_operator


# Gradient

struct GradientOperator{T<:Real,N,M}<:AbstractLinearOperator{T,M,T,M}
    op::AbstractLinearOperator{T,M,T,M}
end

function gradient_operator(h::NTuple{N,T}) where {T<:Real,N}
    padding = Tuple([(0, 0) for i = 1:N])
    stencil_size = Tuple([[2 for i = 1:N]..., 1, N])
    stencil = zeros(T, stencil_size)
    for i = 1:N
        idx = ones(Int, N)
        stencil[idx..., 1, i] = -1/h[i]
        idx[i] = 2
        stencil[idx..., 1, i] =  1/h[i]
    end
    ∇_op = convolution_operator(reverse(stencil), padding)
    return GradientOperator{T,N,N+2}(∇_op)
end

AbstractLinearOperators.domain_size(∇::GradientOperator) = domain_size(∇.op)
AbstractLinearOperators.range_size(∇::GradientOperator) = range_size(∇.op)
AbstractLinearOperators.label(::GradientOperator) = "∇"

AbstractLinearOperators.matvecprod(∇::GradientOperator{T,N,M}, u::AbstractArray{T,M}) where {T,N,M} = matvecprod(∇.op, u)
AbstractLinearOperators.matvecprod_adj(∇::GradientOperator{T,N,M}, u::AbstractArray{T,M}) where {T,N,M} = matvecprod_adj(∇.op, u)

AbstractLinearOperators.to_full_matrix(G::GradientOperator) = to_full_matrix(G.op)