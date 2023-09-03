export ConvolutionOperator, convolution_operator

mutable struct ConvolutionOperator{T,N}<:AbstractLinearOperator{T,N,T,N}
    stencil::AbstractArray{T,N}
    cdims::Union{Nothing,DenseConvDims}
    padding
end

convolution_operator(stencil::AbstractArray{T,N}; padding::Union{Nothing,NTuple{N,NTuple{2,Integer}}}=nothing) where {T,N} = ConvolutionOperator{T,N}(reverse(stencil), nothing, isnothing(padding) ? tuple(zeros(Integer, 2*N)...) : flatten(padding))

AbstractLinearOperators.domain_size(C::ConvolutionOperator) = is_init(C) ? NNlib.input_size(C.cdims) : nothing
AbstractLinearOperators.range_size(C::ConvolutionOperator) = is_init(C) ? NNlib.output_size(C.cdims) : nothing
AbstractLinearOperators.label(::ConvolutionOperator) = "Conv"

function AbstractLinearOperators.matvecprod(C::ConvolutionOperator{T,N}, u::AbstractArray{T,N}) where {T,N}
    if ~is_init(C)
        C.stencil = convert(typeof(u), C.stencil)
        C.cdims = DenseConvDims(reshape_conv(u), reshape_conv(C.stencil); padding=C.padding)
    end
    return dropdims(conv(reshape_conv(u), reshape_conv(C.stencil); pad=C.padding); dims=(N+1,N+2))
end
AbstractLinearOperators.matvecprod_adj(C::ConvolutionOperator{T,N}, v::AbstractArray{T,N}) where {T,N} = dropdims(∇conv_data(reshape_conv(v), reshape_conv(C.stencil), C.cdims); dims=(N+1,N+2))

is_init(C::ConvolutionOperator) = ~isnothing(C.cdims)
reshape_conv(u::AbstractArray) = reshape(u, size(u)..., 1, 1)
flatten(padding::NTuple{N,NTuple{2,Integer}}) where N = Tuple(collect(Iterators.flatten(padding)))