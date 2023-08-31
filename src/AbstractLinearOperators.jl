module AbstractLinearOperators

# Modules
using LinearAlgebra

# Abstract type
include("./abstract_type.jl")
include("./concrete_type.jl")

# Algebra
include("./linear_algebra.jl")

# Basic examples
include("./basic_linear_operators.jl")

# Test
include("./test_utils.jl")

end