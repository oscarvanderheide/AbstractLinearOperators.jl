export IdentityOperator, identity_operator
export ReshapeOperator, reshape_operator


# Identity

struct IdentityOperator{T,N}<:AbstractLinearOperator{T,N,N}
    size::NTuple{<:Any,Int64}
end

domain_size(I::IdentityOperator) = I.size
range_size(I::IdentityOperator) = I.size
matvecprod(::IdentityOperator{T,N}, u::AbstractArray{T,N}) where {T,N} = u
matvecprod_adj(::IdentityOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = v

identity_operator(T::DataType, size::NTuple{N,Int64}) where N = IdentityOperator{T,N}(size)

Flux.gpu(I::IdentityOperator) = I
Flux.cpu(I::IdentityOperator) = I


# Reshaping operator

struct ReshapeOperator{T,N1,N2}<:AbstractLinearOperator{T,N1,N2}
    input_size::NTuple{N1,Int64}
    output_size::NTuple{N2,Int64}
end

domain_size(A::ReshapeOperator) = A.input_size
range_size(A::ReshapeOperator) = A.output_size
matvecprod(A::ReshapeOperator{T,N1,N2}, u::AbstractArray{T,N1}) where {T,N1,N2} = reshape(u, A.output_size)
matvecprod_adj(A::ReshapeOperator{T,N1,N2}, v::AbstractArray{T,N2}) where {T,N1,N2} = reshape(v, A.input_size)

reshape_operator(T::DataType, input_size::NTuple{N1,Int64}, output_size::NTuple{N2,Int64}) where {N1,N2} = ReshapeOperator{T,N1,N2}(input_size, output_size)

Flux.gpu(A::ReshapeOperator) = A
Flux.cpu(A::ReshapeOperator) = A