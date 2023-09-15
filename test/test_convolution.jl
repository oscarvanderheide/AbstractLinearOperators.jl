using AbstractLinearOperators, CUDA, cuDNN, Test
CUDA.allowscalar(false)

T = Float64
input_size = 2^6
st_size = 3
nc_in = 2
nc_out = 3
nb = 4
rtol = T(1e-6)

for N = 1:3

    # Linear operator
    stencil = randn(T, st_size*ones(Integer, N)..., nc_in, nc_out)
    padding = Tuple([rand(0:st_size) for i = 1:2*N])
    C = convolution_operator(stencil; padding=padding)

    # Adjoint test
    u = randn(T, input_size*ones(Integer, N)..., nc_in, nb)
    output_size = size(C*u) # initialize
    v = randn(T, output_size)
    @test adjoint_test(C; input=u, output=v, rtol=rtol)

    # Full-matrix coherence
    Cmat = to_full_matrix(C)
    @test reshape(C*u, :, nb) ≈ Cmat*reshape(u, :, nb) rtol=rtol

    # Linear operator
    C = convolution_operator(stencil; padding=padding)

    # Adjoint test
    u = CUDA.randn(T, input_size*ones(Integer, N)..., nc_in, nb)
    v = CUDA.randn(T, output_size)
    @test adjoint_test(C; input=u, output=v, rtol=rtol)

end