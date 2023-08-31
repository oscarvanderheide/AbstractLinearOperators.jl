using AbstractLinearOperators, Test

@testset "AbstractLinearOperators.jl" begin
    include("./test_linalg.jl")
    include("./test_identity.jl")
    include("./test_reshape.jl")
    include("./test_real2complex.jl")
end