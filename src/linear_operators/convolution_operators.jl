export ConvolutionOperator, convolution_operator


# Convolution operator (batched version)

mutable struct ConvolutionOperator{T,N,Nb}<:AbstractLinearOperator{T,Nb,T,Nb}
    stencil::AbstractArray{T,Nb}
    padding::NTuple{N,NTuple{2,Integer}}
    cdims::Union{Nothing,DenseConvDims}
end

function convolution_operator(stencil::AbstractArray{T,Nb}, padding::Union{Nothing,NTuple{N,NTuple{2,Integer}}}) where {T,N,Nb}
    isnothing(padding) && (padding = Tuple([(0, 0) for i = 1:Nb-2])); Nx = length(padding)
    (Nb !== Nx+2) && throw(ArgumentError("Stencil and padding dimensions are not consistent!"))
    return ConvolutionOperator{T,Nx,Nb}(stencil, padding, nothing)
end

AbstractLinearOperators.domain_size(C::ConvolutionOperator) = is_init(C) ? Tuple([NNlib.input_size(C.cdims)..., NNlib.channels_in(C.cdims), "nb"]) : nothing
AbstractLinearOperators.range_size(C::ConvolutionOperator) = is_init(C) ? Tuple([NNlib.output_size(C.cdims)..., NNlib.channels_out(C.cdims), "nb"]) : nothing
AbstractLinearOperators.label(::ConvolutionOperator) = "Conv"

"""
Size of input is WHCN (width, height(, depth), channel, batch)
"""
function AbstractLinearOperators.matvecprod(C::ConvolutionOperator{T,N,Nb}, u::AbstractArray{T,Nb}) where {T,N,Nb}
    if ~is_init(C)
        C.stencil = convert(typeof(u), C.stencil)
        C.cdims = DenseConvDims(size(u), size(C.stencil); padding=collect(Iterators.flatten(C.padding)))
    end
    return conv(u, C.stencil, C.cdims)
end
AbstractLinearOperators.matvecprod_adj(C::ConvolutionOperator{T,N,Nb}, v::AbstractArray{T,Nb}) where {T,N,Nb} = ∇conv_data(v, C.stencil, C.cdims)

is_init(C::ConvolutionOperator) = ~isnothing(C.cdims)

function to_full_matrix(C::ConvolutionOperator{T,N,Nb}) where {T,N,Nb}

    # Input check
    input_size  = domain_size(C)[1:end-1]
    output_size = range_size(C)[1:end-1]
    isnothing(input_size)  && throw(ArgumentError("The input size must be specified"))
    isnothing(output_size) && throw(ArgumentError("The output size must be specified"))

    # Size
    n_in, chs_in  = input_size[1:end-1],  input_size[end]
    _,    chs_out = output_size[1:end-1], output_size[end]

    # Zero padding matrix
    Pm = to_full_matrix(zero_padding_operator(T, C.padding), n_in)
    padded_size = extended_size(n_in, C.padding)

    # Setting diagonal positions
    w = reverse(C.stencil; dims=Tuple(1:N))
    ssize = size(w)[1:N]
    scenter = Tuple(ones(Integer, N))
    ncprod = cumprod((1, padded_size[1:end-1]...))
    diag_pos = similar(w, Int, ssize)
    for cidx = CartesianIndices(ssize)
        diag_pos[cidx] = sum((Tuple(cidx).-scenter).*ncprod)
    end

    # Valid-convolution restriction
    padding_conv = Tuple([(0, ssize[i]-1) for i = 1:N])
    Qm = to_full_matrix(zero_padding_operator(T, padding_conv), padded_size.-sum.(padding_conv))

    # Assembly sparse matrix
    m = prod(padded_size)
    return hvcat(chs_in, [Qm'*spdiagm(m, m, [diag_pos[cidx] => w[cidx, cin, cout]*ones(T, m-abs(diag_pos[cidx]))
                                             for cidx = CartesianIndices(ssize)]...)*Pm
                          for cin = 1:chs_in, cout = 1:chs_out]...)

end