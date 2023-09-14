export WaveletReshape2D, wavelet_reshape_2D
export HaarTransform2D, Haar_transform_2D


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


# # Haar transform (2D)

# mutable struct HaarTransform2D{T}<:AbstractLinearOperators{T,4,T,4}
#     op::AbstractLinearOperator{T,4,T,4}
# end

# function Haar_transform_2D(T::DataType)
#     stencil = cat([ T(0.5)  T(0.5);  T(0.5) T(0.5)], [-T(0.5) -T(0.5);  T(0.5) T(0.5)],
#                   [-T(0.5)  T(0.5); -T(0.5) T(0.5)], [ T(0.5) -T(0.5); -T(0.5) T(0.5)]; dims=4)
#     return HaarTransform2D{T,N}(convolution_operator(stencil, (0,0,0,0)))
# end

# AbstractLinearOperators.domain_size(W::HaarTransform2D{T}) = domain_size(W.op)
# AbstractLinearOperators.range_size(W::HaarTransform2D{T}) = domain_size(W.op)
# AbstractLinearOperators.label(::HaarTransform2D{T}) = "HaarTransform2D"

# function AbstractLinearOperators.matvecprod(W::HaarTransform2D{T}, u::AbstractArray{T,4}) where {T}
#     nx, ny = size(u,1), size(u,2)
#     n_lvl = _maxtransformlevels(u)
#     Wu = similar(u)
#     @inbounds for l = 1:n_lvl
#         idx_l = (1:div(nx,2^(l-1)), 1:div(ny,2^(l-1)))
#         Wu[idx_l..., :, :] .= multich2im_reshape(W.op*u[idx_l..., :, :])
#     end
#     return Wu
# end

# function AbstractLinearOperators.matvecprod_adj(W::HaarTransform2D{T}, v::AbstractArray{T,4}) where {T,N}
#     nx, ny = size(v,1), size(v,2)
#     n_lvl = _maxtransformlevels(v)
#     WTv = similar(v)
#     @inbounds for l = n_lvl:-1:1
#         idx_l = (1:div(nx,2^(l-1)), 1:div(ny,2^(l-1)))
#         WTv[idx_l..., :, :] .= W.op'*wavelet_reshape(v[idx_l..., :, :])
#     end
#     return WTv
# end


# # Haar utils

# _maxtransformlevels(x::AbstractArray) = minimum(_maxtransformlevels.(size(x)))

# function _maxtransformlevels(arraysize::Integer)
#     arraysize > 1 || return 0
#     tl = 0
#     while _sufficientpoweroftwo(arraysize, tl)
#         tl += 1
#     end
#     return tl - 1
# end

# function _sufficientpoweroftwo(x::AbstractArray, L::Integer)
#     for i = 1:ndims(x)
#         _sufficientpoweroftwo(size(x,i), L) || return false
#     end
#     return true
# end
# _sufficientpoweroftwo(n::Integer, L::Integer) = (n%(2^L) == 0)