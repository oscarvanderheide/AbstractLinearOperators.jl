module AbstractLinearOperators

# Modules
using LinearAlgebra#, Flux
import Base: size, show, eltype, +, -, *, /, \
import LinearAlgebra: adjoint, inv

RealOrComplex{T<:Real} = Union{T, Complex{T}}

# Abstract type
include("./abstract_type.jl")
include("./concrete_type.jl")

# Algebra
include("./linear_algebra.jl")

# Examples
include("./examples.jl")

end