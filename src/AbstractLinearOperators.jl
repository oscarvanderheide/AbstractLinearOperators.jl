module AbstractLinearOperators

# Modules
using LinearAlgebra
import Base: size, show, eltype, +, -, *, /
import LinearAlgebra: adjoint

# Abstract type
include("./abstract_type.jl")
include("./concrete_type.jl")

# Utils
include("./utils.jl")

# Algebra
include("./linear_algebra.jl")

end
