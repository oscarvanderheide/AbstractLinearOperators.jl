using AbstractLinearOperators, CUDA, cuDNN, Test
CUDA.allowscalar(false)

# Linear operator
T = Float64
input_size = (2^7, 2^8)
stencil = randn(T, 3, 2)
C = convolution_operator(stencil)
u = randn(T, input_size); C*u # initialize

# Adjoint test
rtol = T(1e-6)
@test adjoint_test(C; rtol=rtol)

# Linear operator
stencil = CUDA.randn(T, 3, 2)
C = convolution_operator(stencil)
u = CUDA.randn(T, input_size)
output_size = size(C*u) # initialize

# Adjoint test
rtol = T(1e-6)
u = CUDA.randn(T, input_size)
v = CUDA.randn(T, output_size)
@test adjoint_test(C; input=u, output=v, rtol=rtol)