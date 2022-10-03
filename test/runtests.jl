using AbstractLinearOperators, Test

@testset "AbstractLinearOperators.jl" begin
    include("./test_linalg.jl")
    include("./test_reshape.jl")
end