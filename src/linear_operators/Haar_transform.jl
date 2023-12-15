export HaarTransform, Haar_transform


# Haar transform

mutable struct HaarTransform{T,N,Nb}<:AbstractLinearOperator{T,Nb,T,Nb}
    op::ConvolutionOperator{T,N,Nb}
    factor::Union{Nothing,T}
end

Haar_transform(::Type{T}, N::Integer; orthogonal::Bool=true) where T = HaarTransform{T,N,N+2}(convolution_operator(Haar_stencil(T, N); stride=2, flipped=true), orthogonal ? nothing : (sqrt(T(2))/2)^N)

AbstractLinearOperators.domain_size(W::HaarTransform) = domain_size(W.op)
AbstractLinearOperators.range_size(W::HaarTransform)  = range_size(W.op)
AbstractLinearOperators.label(::HaarTransform) = "HaarTransform"

function AbstractLinearOperators.matvecprod(W::HaarTransform{T,N,Nb}, u::AbstractArray{T,Nb}; inverse::Bool=false) where {T,N,Nb}

    n..., nc, nb = size(u); nc_ = nc*2^N
    Wu = similar(u, div.(n,2)..., nc_, nb)
    idx_spatial = Tuple([Colon() for i = 1:length(n)])
    @inbounds for c = 1:nc
        selectdim(Wu, N+1, c:nc:nc_) .= matvecprod(W.op, u[idx_spatial...,c:c,:])
    end
    ~isnothing(W.factor) && (~inverse ? (Wu .*= W.factor) : (Wu ./= W.factor))
    return Wu

end

function AbstractLinearOperators.matvecprod_adj(W::HaarTransform{T,N,Nb}, Wu::AbstractArray{T,Nb}; inverse::Bool=false) where {T,N,Nb}

    n..., nc_, nb = size(Wu); nc = div(nc_,2^N)
    u = similar(Wu, n.*2..., nc, nb)
    idx_spatial = Tuple([Colon() for i = 1:length(n)])
    @inbounds for c = 1:nc
        selectdim(u, N+1, c:c) .= matvecprod_adj(W.op, Wu[idx_spatial...,c:nc:nc_,:])
    end
    ~isnothing(W.factor) && (~inverse ? (u .*= W.factor) : (u ./= W.factor))
    return u

end

AbstractLinearOperators.invmatvecprod(W::HaarTransform{T,N,Nb}, v::AbstractArray{T,Nb}) where {T,N,Nb} = matvecprod_adj(W, v; inverse=true)
AbstractLinearOperators.invmatvecprod_adj(W::HaarTransform{T,N,Nb}, u::AbstractArray{T,Nb}) where {T,N,Nb} = matvecprod(W, u; inverse=true)


# Haar utils

function Haar_stencil(::Type{T}, N::Integer) where T

    # One-dimensional case
    (N == 1) && (return cat([1; 1], [-1; 1]; dims=3)/sqrt(T(2)))

    # Initialize stencil
    H = Array{T,N+2}(undef,2*ones(Int,N)...,1,2^N)

    # Recursive definition
    H_1   = Haar_stencil(T, 1)
    H_Nm1 = Haar_stencil(T, N-1)
    @inbounds for i = 1:2^(N-1), j = 1:2
        selectdim(selectdim(H, N+2, 2^(N-1)*(j-1)+i), N+1, 1) .= selectdim(selectdim(H_Nm1, N+1, i), N, 1).*reshape(H_1[:,1,j],ones(Int,N-1)...,2)
    end

    return H

end