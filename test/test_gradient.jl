using AbstractLinearOperators, CUDA, cuDNN, Test
CUDA.allowscalar(false)

T = Float64
input_size = 2^6
rtol = T(1e-6)
nb = 2

for N = [2, 3]

    # Gradient operator
    h = Tuple(abs.(randn(T, N)))
    ∇ = gradient_operator(h)

    # Adjoint test
    sz = input_size*ones(Integer, N)
    u = randn(T, Tuple(sz)..., 1, nb); output_size = size(∇*u)
    v = randn(T, output_size)
    @test adjoint_test(∇; input=u, output=v, rtol=rtol)

end

for N = [2, 3]

    # Gradient linear operator
    h = Tuple(abs.(randn(T, N)))
    ∇ = gradient_operator(h)

    # Adjoint test
    sz = input_size*ones(Integer, N)
    u = CUDA.randn(T, Tuple(sz)..., 1, nb); output_size = size(∇*u)
    v = CUDA.randn(T, output_size)
    @test adjoint_test(∇; input=u, output=v, rtol=rtol)

end

# Matrix coherency
for N = [2, 3]

    # Gradient operator
    h = Tuple(abs.(randn(T, N)))
    ∇ = gradient_operator(h)

    # Coherence test
    sz = input_size*ones(Integer, N)
    u = randn(T, Tuple(sz)..., 1, nb); output_size = size(∇*u)
    ∇m = to_full_matrix(∇)
    @test ∇m*reshape(u, :, nb) ≈ reshape(∇*u, :, nb) rtol=rtol

end