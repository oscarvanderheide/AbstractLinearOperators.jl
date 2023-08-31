export IdentityOperator, identity_operator,
       ReshapeOperator, reshape_operator,
       Real2ComplexOperator, real2complex_operator, Complex2RealOperator, complex2real_operator,
       AbstractPaddingOperator, ZeroPaddingOperator, zero_padding_operator


# Identity

struct IdentityOperator{T,N}<:AbstractLinearOperator{T,N,T,N}
    size::Union{Nothing, NTuple{N,Integer}}
end

identity_operator(T::DataType; size::Union{Nothing, NTuple{N,Integer}}=nothing) where N = IdentityOperator{T,N}(size)

domain_size(I::IdentityOperator) = I.size
range_size(I::IdentityOperator) = I.size
label(::IdentityOperator) = "Id"
matvecprod(::IdentityOperator{T,N}, u::AbstractArray{T,N}) where {T,N} = u
matvecprod_adj(::IdentityOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = v
invmatvecprod(::IdentityOperator{T,N}, u::AbstractArray{T,N}) where {T,N} = u
invmatvecprod_adj(::IdentityOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = v


# Reshaping operator

struct ReshapeOperator{T,N1,N2}<:AbstractLinearOperator{T,N1,T,N2}
    input_size::NTuple{N1,Integer}
    output_size::NTuple{N2,Integer}
end

reshape_operator(T::DataType, input_size::NTuple{N1,Integer}, output_size::NTuple{N2,Integer}) where {N1,N2} = ReshapeOperator{T,N1,N2}(input_size, output_size)

domain_size(A::ReshapeOperator) = A.input_size
range_size(A::ReshapeOperator) = A.output_size
label(A::ReshapeOperator) = "Reshape"
matvecprod(A::ReshapeOperator{T,N1,N2}, u::AbstractArray{T,N1}) where {T,N1,N2} = reshape(u, A.output_size)
matvecprod_adj(A::ReshapeOperator{T,N1,N2}, v::AbstractArray{T,N2}) where {T,N1,N2} = reshape(v, A.input_size)


# Real to complex & viceversa

struct Real2ComplexOperator{T<:Real,N}<:AbstractLinearOperator{T,N,Complex{T},N}
    size::Union{Nothing, NTuple{N,Integer}}
end

real2complex_operator(T::DataType; size::Union{Nothing, NTuple{N,Integer}}=nothing) where N = T<:Real ? Real2ComplexOperator{T,N}(size) : error("Element type must be real")

domain_size(A::Real2ComplexOperator) = A.size
range_size(A::Real2ComplexOperator) = A.size
label(::Real2ComplexOperator) = "Complex"
matvecprod(::Real2ComplexOperator{T,N}, u::AbstractArray{T,N}) where {T,N} = complex(u)
matvecprod_adj(::Real2ComplexOperator{T,N}, v::AbstractArray{Complex{T},N}) where {T,N} = real(v)

struct Complex2RealOperator{T<:Real,N}<:AbstractLinearOperator{Complex{T},N,T,N}
    size::Union{Nothing, NTuple{N,Integer}}
end

complex2real_operator(T::DataType; size::Union{Nothing, NTuple{N,Integer}}=nothing) where N = T<:Complex ? Complex2RealOperator{real(T),N}(size) : error("Element type must be complex")

domain_size(A::Complex2RealOperator) = A.size
range_size(A::Complex2RealOperator) = A.size
label(::Complex2RealOperator) = "Real"
matvecprod(::Complex2RealOperator{T,N}, u::AbstractArray{Complex{T},N}) where {T,N} = real(u)
matvecprod_adj(::Complex2RealOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = complex(v)


# Padding operators

abstract type AbstractPaddingOperator{T,N}<:AbstractLinearOperator{T,N,T,N} end

struct ZeroPaddingOperator{T,N}<:AbstractPaddingOperator{T,N}
    size::NTuple{N,Integer}
    padding::NTuple{N, NTuple{2,Integer}}
    extended_size::NTuple{N,Integer}
    center_view
end

function zero_padding_operator(T::DataType, size::NTuple{N,Integer}, padding::NTuple{N, NTuple{2,Integer}}) where N
    ext_size = extended_size(size, padding)
    cv = center_view(ext_size, padding)
    return ZeroPaddingOperator{T,N}(size, padding, ext_size, cv)
end

domain_size(P::ZeroPaddingOperator) = P.size
range_size(P::ZeroPaddingOperator) = P.extended_size
label(::AbstractPaddingOperator) = "Padding"

function matvecprod(P::ZeroPaddingOperator{T,N}, u::AbstractArray{T,N}) where {T,N}
    u_ext = zeros(T, P.extended_size)
    view(u_ext, P.center_view...) .= u
    return u_ext
end

matvecprod_adj(P::ZeroPaddingOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = v[P.center_view...]

extended_size(size::NTuple{N,Integer}, padding::NTuple{N, NTuple{2,Integer}}) where N = size.+sum.(padding)
center_view(ext_size::NTuple{N,Integer}, padding::NTuple{N, NTuple{2,Integer}}) where N = Tuple([padding[i][1]+1:ext_size[i]-padding[i][2] for i = 1:N])