export ConvolutionOperator, convolution_operator


# Convolution operator (batched version)

mutable struct ConvolutionOperator{T,N,Nb}<:AbstractLinearOperator{T,Nb,T,Nb}
    stencil::AbstractArray{T,Nb}
    padding::NTuple{N,NTuple{2,Integer}}
    cdims::Union{Nothing,DenseConvDims}
    batch_size::Union{Nothing,Integer}
end

function convolution_operator(stencil::AbstractArray{T,Nb}, padding::Union{Nothing,NTuple{N,NTuple{2,Integer}}}) where {T,N,Nb}
    isnothing(padding) && (padding = Tuple([(0, 0) for i = 1:Nb-2])); Nx = length(padding)
    (Nb !== Nx+2) && throw(ArgumentError("Stencil and padding dimensions are not consistent!"))
    return ConvolutionOperator{T,Nx,Nb}(stencil, padding, nothing, nothing)
end

AbstractLinearOperators.domain_size(C::ConvolutionOperator) = is_init(C) ? Tuple([NNlib.input_size(C.cdims)..., NNlib.channels_in(C.cdims), C.batch_size]) : nothing
AbstractLinearOperators.range_size(C::ConvolutionOperator) = is_init(C) ? Tuple([NNlib.output_size(C.cdims)..., NNlib.channels_out(C.cdims), C.batch_size]) : nothing
AbstractLinearOperators.label(::ConvolutionOperator) = "Conv"

"""
Size of input is WHCN (width, height(, depth), channel, batch)
"""
function AbstractLinearOperators.matvecprod(C::ConvolutionOperator{T,N,Nb}, u::AbstractArray{T,Nb}) where {T,N,Nb}
    if ~is_init(C)
        C.stencil = convert(typeof(u), C.stencil)
        C.cdims = DenseConvDims(size(u), size(C.stencil); padding=collect(Iterators.flatten(C.padding)))
        C.batch_size = size(u, Nb)
    end
    return conv(u, C.stencil, C.cdims)
end
AbstractLinearOperators.matvecprod_adj(C::ConvolutionOperator{T,N,Nb}, v::AbstractArray{T,Nb}) where {T,N,Nb} = ∇conv_data(v, C.stencil, C.cdims)

is_init(C::ConvolutionOperator) = ~isnothing(C.cdims)