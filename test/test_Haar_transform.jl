using AbstractLinearOperators, CUDA, cuDNN, Test
CUDA.allowscalar(false)

T = Float64
input_size = 2^6
nc = 3
nb = 4
rtol = T(1e-6)

println("Test Haar transform")
for N = 1:3, orthogonal = [true, false], device = [:gpu, :cpu]
    println("N=", N, "; orthogonal=", orthogonal, "; device=", device)

    # Linear operator
    W = Haar_transform(T, N; orthogonal=orthogonal)

    # Random input
    n = input_size*ones(Integer, N)
    u = randn(T, n..., nc, nb); (device == :gpu) && (u = CuArray(u))
    v = randn(T, div.(n, 2)..., nc*2^N, nb); (device == :gpu) && (v = CuArray(v))

    # Adjoint test
    @test adjoint_test(W; input=u, output=v, rtol=rtol)

    # Inverse test
    @test inverse_test(W; input=u, output=v, rtol=rtol)

end