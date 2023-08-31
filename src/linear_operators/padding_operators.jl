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

function matvecprod(P::RepeatPaddingOperator{T,2}, u::AbstractArray{T,2}) where T

    u_ext = similar(u, P.extended_size)
    view(u_ext, P.center_view...) .= u
    u_ext[1:P.padding[1][1], P.padding[2][1]+1:end-P.padding[2][2]] .= u_ext[P.padding[1][1]+1:P.padding[1][1]+1, P.padding[2][1]+1:end-P.padding[2][2]]
    u_ext[end-P.padding[1][2]+1:end, P.padding[2][1]+1:end-P.padding[2][2]] .= u_ext[end-P.padding[1][2]:end-P.padding[1][2], P.padding[2][1]+1:end-P.padding[2][2]]
    u_ext[:, 1:P.padding[2][1]] .= u_ext[:, P.padding[2][1]+1:P.padding[2][1]+1]
    u_ext[:, end-P.padding[2][2]+1:end] .= u_ext[:, end-P.padding[2][2]:end-P.padding[2][2]]
    return u_ext

end

function matvecprod_adj(P::RepeatPaddingOperator{T,2}, u_ext::AbstractArray{T,2}) where T

    padding = P.padding
    u = copy(view(u_ext, P.center_view...))

    u_ext_top = sum(view(u_ext, 1:padding[1][1], :); dims=1)
    u[1:1, :] .+= [sum(u_ext_top[:, 1:padding[2][1]+1]; dims=2) u_ext_top[:, padding[2][1]+2:end-padding[2][2]-1] sum(u_ext_top[:, end-padding[2][2]:end]; dims=2)]

    u_ext_bottom = sum(view(u_ext, P.extended_size[1]-padding[1][2]+1:P.extended_size[1], :); dims=1)
    u[end:end, :] .+= [sum(u_ext_bottom[:, 1:padding[2][1]+1]; dims=2) u_ext_bottom[:, padding[2][1]+2:end-padding[2][2]-1] sum(u_ext_bottom[:, end-padding[2][2]:end]; dims=2)]

    u_ext_left = sum(view(u_ext, padding[1][1]+1:P.extended_size[1]-padding[1][2], 1:padding[2][1]); dims=2)
    u[:, 1:1] .+= u_ext_left

    u_ext_right = sum(view(u_ext, padding[1][1]+1:P.extended_size[1]-padding[1][2], P.extended_size[2]-padding[2][2]+1:P.extended_size[2]); dims=2)
    u[:, end:end] .+= u_ext_right

    return u

end