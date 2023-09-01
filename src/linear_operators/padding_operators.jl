export AbstractPaddingOperator,
       ZeroPaddingOperator, zero_padding_operator,
       RepeatPaddingOperator, repeat_padding_operator,
       extended_size


# Padding operators

abstract type AbstractPaddingOperator{T,N}<:AbstractLinearOperator{T,N,T,N} end

struct ZeroPaddingOperator{T,N}<:AbstractPaddingOperator{T,N}
    padding::NTuple{N, NTuple{2,Integer}}
end

zero_padding_operator(T::DataType, padding::NTuple{N, NTuple{2,Integer}}) where N = ZeroPaddingOperator{T,N}(padding)

label(::AbstractPaddingOperator) = "Padding"

function matvecprod(P::ZeroPaddingOperator{T,N}, u::AbstractArray{T,N}) where {T,N}
    ext_size = extended_size(size(u), P.padding)
    cview = center_view(ext_size, P.padding)
    u_ext = zeros(T, ext_size)
    view(u_ext, cview...) .= u
    return u_ext
end

function matvecprod_adj(P::ZeroPaddingOperator{T,N}, u_ext::AbstractArray{T,N}) where {T,N}
    ext_size = size(u_ext)
    cview = center_view(ext_size, P.padding)
    return u_ext[cview...]
end

extended_size(size::NTuple{N,Integer}, padding::NTuple{N, NTuple{2,Integer}}) where N = size.+sum.(padding)
center_view(ext_size::NTuple{N,Integer}, padding::NTuple{N, NTuple{2,Integer}}) where N = Tuple([padding[i][1]+1:ext_size[i]-padding[i][2] for i = 1:N])

struct RepeatPaddingOperator{T,N}<:AbstractPaddingOperator{T,N}
    padding::NTuple{N, NTuple{2,Integer}}
end

repeat_padding_operator(T::DataType, padding::NTuple{N, NTuple{2,Integer}}) where N = RepeatPaddingOperator{T,N}(padding)

function matvecprod(P::RepeatPaddingOperator{T,N}, u::AbstractArray{T,N}) where {T,N}

    padding = P.padding
    ext_size = extended_size(size(u), P.padding)
    cview = center_view(ext_size, P.padding)
    u_ext = similar(u, ext_size)
    view(u_ext, cview...) .= u
    @inbounds for i = 1:N
        selectdim(u_ext, i, 1:padding[i][1]) .= selectdim(u_ext, i, padding[i][1]+1:padding[i][1]+1)
        selectdim(u_ext, i, ext_size[i]-padding[i][2]+1:ext_size[i]) .= selectdim(u_ext, i, ext_size[i]-padding[i][2]:ext_size[i]-padding[i][2])
    end
    return u_ext

end

function matvecprod_adj(P::RepeatPaddingOperator{T,N}, u_ext::AbstractArray{T,N}) where {T,N}

    padding = P.padding
    ext_size = size(u_ext)
    u = copy(u_ext)
    @inbounds for i = 1:N
        u = cat(sum(selectdim(u, i, 1:padding[i][1]+1); dims=i),
                selectdim(u, i, padding[i][1]+2:ext_size[i]-padding[i][2]-1),
                sum(selectdim(u, i, ext_size[i]-padding[i][2]:ext_size[i]); dims=i);
                dims=i)
    end
    return u

end