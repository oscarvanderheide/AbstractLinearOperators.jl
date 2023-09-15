using AbstractLinearOperators, CUDA, cuDNN, Test
CUDA.allowscalar(false)

T = Float64
N = 2
input_size = 2^6
st_size = 3
nc = 8
nb = 4
rtol = T(1e-6)

# Linear operator
R = wavelet_reshape_2D(T)

# Adjoint test
u = randn(T, input_size*ones(Integer, N)..., nc, nb)
v = randn(T, 2*input_size*ones(Integer, N)..., div(nc,4), nb)
@test adjoint_test(R; input=u, output=v, rtol=rtol)

# Inverse test
@test inverse_test(R; input=u, output=v, rtol=rtol)

# Linear operator
R = wavelet_reshape_2D(T)

# Adjoint test
u = CUDA.randn(T, input_size*ones(Integer, N)..., nc, nb)
v = CUDA.randn(T, 2*input_size*ones(Integer, N)..., div(nc,4), nb)
@test adjoint_test(R; input=u, output=v, rtol=rtol)

# Inverse test
@test inverse_test(R; input=u, output=v, rtol=rtol)