using AbstractLinearOperators, CUDA, cuDNN, Test
CUDA.allowscalar(false)

T = Float64
N = 2
input_size = 2^6
st_size = 3
nc_in = 2
nc_out = 3
nb = 4
rtol = T(1e-6)

# Linear operator
W = Haar_transform_2D(T)

# Adjoint test
u = randn(T, input_size*ones(Integer, N)..., nc_in, nb)
v = randn(T, input_size*ones(Integer, N)..., nc_in, nb)
@test adjoint_test(W; input=u, output=v, rtol=rtol)

# Full-matrix coherence
Wm = to_full_matrix(W)
@test reshape(W*u, :, nb) ≈ Wm*reshape(u, :, nb) rtol=rtol

# Linear operator
W = Haar_transform_2D(T)

# Adjoint test
u = CUDA.randn(T, input_size*ones(Integer, N)..., nc_in, nb)
v = CUDA.randn(T, input_size*ones(Integer, N)..., nc_in, nb)
@test adjoint_test(W; input=u, output=v, rtol=rtol)