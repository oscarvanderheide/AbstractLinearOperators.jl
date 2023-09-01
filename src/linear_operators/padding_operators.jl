export AbstractPaddingOperator, ZeroPaddingOperator, zero_padding_operator, RepeatPaddingOperator, repeat_padding_operator


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

struct RepeatPaddingOperator{T,N}<:AbstractPaddingOperator{T,N}
    size::NTuple{N,Integer}
    padding::NTuple{N, NTuple{2,Integer}}
    extended_size::NTuple{N,Integer}
    center_view
end

function repeat_padding_operator(T::DataType, size::NTuple{N,Integer}, padding::NTuple{N, NTuple{2,Integer}}) where N
    ext_size = extended_size(size, padding)
    cv = center_view(ext_size, padding)
    return RepeatPaddingOperator{T,N}(size, padding, ext_size, cv)
end

domain_size(P::RepeatPaddingOperator) = P.size
range_size(P::RepeatPaddingOperator) = P.extended_size

function matvecprod(P::RepeatPaddingOperator{T,N}, u::AbstractArray{T,N}) where {T,N}

    u_ext = similar(u, P.extended_size)
    view(u_ext, P.center_view...) .= u
    @inbounds for i = 1:N
        selectdim(u_ext, i, 1:P.padding[i][1]) .= selectdim(u_ext, i, P.padding[i][1]+1:P.padding[i][1]+1)
        selectdim(u_ext, i, P.extended_size[i]-P.padding[i][2]+1:P.extended_size[i]) .= selectdim(u_ext, i, P.extended_size[i]-P.padding[i][2]:P.extended_size[i]-P.padding[i][2])
    end
    return u_ext

end

function matvecprod_adj(P::RepeatPaddingOperator{T,N}, u_ext::AbstractArray{T,N}) where {T,N}

    padding = P.padding
    ext_size = P.extended_size
    u = copy(u_ext)
    @inbounds for i = 1:N
        u = cat(sum(selectdim(u, i, 1:padding[i][1]+1); dims=i),
                selectdim(u, i, padding[i][1]+2:ext_size[i]-padding[i][2]-1),
                sum(selectdim(u, i, ext_size[i]-padding[i][2]:ext_size[i]); dims=i);
                dims=i)
    end
    return u

end