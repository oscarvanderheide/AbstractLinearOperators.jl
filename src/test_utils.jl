#: Utilities for testing operators

export adjoint_test, inverse_test

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

function adjoint_test(A::AbstractLinearOperator{CTD,ND,CTR,NR};
                      input::Union{Nothing,AbstractArray{CTD,ND}}=nothing,
                      output::Union{Nothing,AbstractArray{CTR,NR}}=nothing,
                      rtol::Union{Nothing,TD,TR}=nothing) where {TD<:AbstractFloat,ND,TR<:AbstractFloat,NR,CTD<:RealOrComplex{TD},CTR<:RealOrComplex{TR}}

    isnothing(input)  && (input  = randn(CTD, domain_size(A)))
    isnothing(output) && (output = randn(CTR, range_size(A)))
    isnothing(rtol) && (rtol = findmax((eps(TD), eps(TR)))[1])
    return isapprox(real(dot(A*input, output)), real(dot(input, adjoint(A)*output)); rtol=rtol)

end

function inverse_test(A::AbstractLinearOperator{CTD,ND,CTR,NR};
                      input::Union{Nothing,AbstractArray{CTD,ND}}=nothing,
                      output::Union{Nothing,AbstractArray{CTR,NR}}=nothing,
                      rtol::Union{Nothing,TD,TR}=nothing) where {TD<:AbstractFloat,ND,TR<:AbstractFloat,NR,CTD<:RealOrComplex{TD},CTR<:RealOrComplex{TR}}

    isnothing(input)  && (input  = randn(CTD, domain_size(A)))
    isnothing(output) && (output = randn(CTR, range_size(A)))
    isnothing(rtol) && (rtol = findmax((eps(TD), eps(TR)))[1])
    Ainv = inv(A)
    return isapprox(Ainv*(A*input), input; rtol=rtol) &&
           isapprox(A*(Ainv*output), output; rtol=rtol)

end