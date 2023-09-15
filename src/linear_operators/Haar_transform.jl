export AbstractWaveletTransform
export HaarTransform2D, Haar_transform_2D
export WaveletReshape2D, wavelet_reshape_2D

abstract type AbstractWaveletTransform{T1,N1,T2,N2}<:AbstractLinearOperator{T1,N1,T2,N2} end


# Wavelet reshape operator (multi-channel to image flattening)

struct WaveletReshape2D{T}<:AbstractLinearOperator{T,4,T,4} end

wavelet_reshape_2D(T::DataType) = WaveletReshape2D{T}()

AbstractLinearOperators.label(::WaveletReshape2D) = "WaveletReshape2D"

function AbstractLinearOperators.matvecprod(::WaveletReshape2D{T}, u::AbstractArray{T,4}) where {T}
    nx, ny, nc, nb = size(u)
    nx_ = 2*nx; ny_ = 2*ny; nc_ = div(nc,4)
    ū = similar(u, T, nx_, ny_, nc_, nb)
    @inbounds for i = 1:2, j = 1:2
        k = i+2*(j-1)
        ū[(i-1)*nx+1:i*nx,(j-1)*ny+1:j*ny,:,:] .= u[:,:,(k-1)*nc_+1:k*nc_,:]
    end
    return ū
end

function AbstractLinearOperators.matvecprod_adj(::WaveletReshape2D{T}, ū::AbstractArray{T,4}) where {T}
    nx_, ny_, nc_, nb = size(ū)
    nx = div(nx_,2); ny = div(ny_,2); nc = nc_*4
    u = similar(ū, T, nx, ny, nc, nb)
    @inbounds for i = 1:2, j = 1:2
        k = i+2*(j-1)
        u[:,:,(k-1)*nc_+1:k*nc_,:] .= ū[(i-1)*nx+1:i*nx,(j-1)*ny+1:j*ny,:,:]
    end
    return u
end

AbstractLinearOperators.invmatvecprod(R::WaveletReshape2D{T}, ū::AbstractArray{T,4}) where {T} = matvecprod_adj(R, ū)
AbstractLinearOperators.invmatvecprod_adj(R::WaveletReshape2D{T}, u::AbstractArray{T,4}) where {T} = matvecprod(R, u)


# Haar transform (2D)

mutable struct HaarTransform2D{T}<:AbstractWaveletTransform{T,4,T,4}
    op::ConvolutionOperator{T,2,4}
    flatten::WaveletReshape2D{T}
    n_levels::Union{Nothing,Integer}
    cdims_level
    idx_level
end

function Haar_transform_2D(T::DataType; n_levels::Union{Nothing,Integer}=nothing)
    stencil = cat([ T(0.5)  T(0.5);  T(0.5) T(0.5)], [-T(0.5) -T(0.5);  T(0.5) T(0.5)], [-T(0.5)  T(0.5); -T(0.5) T(0.5)], [ T(0.5) -T(0.5); -T(0.5) T(0.5)]; dims=4)
    return HaarTransform2D{T}(convolution_operator(stencil; stride=2, flipped=true), wavelet_reshape_2D(T), n_levels, nothing, nothing)
end

AbstractLinearOperators.domain_size(W::HaarTransform2D) = ~is_init(W) ? nothing : input_dims(W.cdims_level[1])
AbstractLinearOperators.range_size(W::HaarTransform2D)  = ~is_init(W) ? nothing : output_dims(W.cdims_level[1])
AbstractLinearOperators.label(::HaarTransform2D) = "HaarTransform2D"

function AbstractLinearOperators.matvecprod(W::HaarTransform2D{T}, u::AbstractArray{T,4}) where {T}
    ~is_init(W) && initialize!(W, u)
    Wu = copy(u)
    @inbounds for l = 1:n_levels(W)
        set_cdims!(W.op, W.cdims_level[l])
        Wu[W.idx_level[l]..., :, :] .= W.flatten*(W.op*Wu[W.idx_level[l]..., :, :])
    end
    return Wu
end

function AbstractLinearOperators.matvecprod_adj(W::HaarTransform2D{T}, v::AbstractArray{T,4}) where {T}
    WTv = copy(v)
    @inbounds for l = n_levels(W):-1:1
        set_cdims!(W.op, W.cdims_level[l])
        WTv[W.idx_level[l]..., :, :] .= W.op'*(W.flatten'*WTv[W.idx_level[l]..., :, :])
    end
    return WTv
end

AbstractLinearOperators.invmatvecprod(W::HaarTransform2D{T}, v::AbstractArray{T,4}) where {T} = matvecprod_adj(W, v)
AbstractLinearOperators.invmatvecprod_adj(W::HaarTransform2D{T}, u::AbstractArray{T,4}) where {T} = matvecprod(W, u)

is_init(W::HaarTransform2D) = ~isnothing(W.cdims_level)
function initialize!(W::HaarTransform2D{T}, u::AbstractArray{T,4}) where {T}
    initialize!(W.op, u)
    nx, ny, _, nb = size(u)
    n_levels = _maxtransformlevels(u)
    if isnothing(W.n_levels)
        W.n_levels = n_levels
    else
        n_levels < W.n_levels ? (@warn "The number of transform levels specified exceeds the maximum. Resetting..."; W.n_levels = n_levels) : (n_levels = W.n_levels)
    end
    cdims_level = Vector{DenseConvDims}(undef, n_levels)
    idx_level = Vector{NTuple{2,UnitRange{Int}}}(undef, n_levels)
    @inbounds for l = 1:n_levels
        nx_l = div(nx,2^(l-1)); ny_l = div(ny,2^(l-1))
        idx_level[l] =  (1:nx_l, 1:ny_l)
        cdims_level[l] = DenseConvDims((nx_l, ny_l, 1, nb), (2, 2, 1, 4); stride=2)
    end
    W.cdims_level = Tuple(cdims_level)
    W.idx_level = Tuple(idx_level)
end
n_levels(W) = length(W.cdims_level)


# Haar utils

_maxtransformlevels(x::AbstractArray{T,4}) where T = minimum(_maxtransformlevels.(size(x)[1:2]))

function _maxtransformlevels(arraysize::Integer)
    arraysize > 1 || return 0
    tl = 0
    while _sufficientpoweroftwo(arraysize, tl)
        tl += 1
    end
    return tl - 1
end

function _sufficientpoweroftwo(x::AbstractArray, L::Integer)
    for i = 1:ndims(x)
        _sufficientpoweroftwo(size(x,i), L) || return false
    end
    return true
end
_sufficientpoweroftwo(n::Integer, L::Integer) = (n%(2^L) == 0)