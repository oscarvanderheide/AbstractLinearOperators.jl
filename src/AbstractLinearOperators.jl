module AbstractLinearOperators

# Modules
using LinearAlgebra, NNlib

# Abstract type
include("./abstract_type.jl")
include("./concrete_type.jl")

# Algebra
include("./linear_algebra.jl")

# Basic examples
include("./linear_operators/basic_linear_operators.jl")
include("./linear_operators/padding_operators.jl")
include("./linear_operators/convolution_operators.jl")

# Test
include("./test_utils.jl")

end