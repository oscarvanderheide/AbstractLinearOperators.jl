using AbstractLinearOperators, Test

# Linear operator
input_size = (2^7, 2^8)
padding = ((1, 2), (3, 4))
T = Float64
A = zero_padding_operator(T, padding)

# Zero padding test
rtol = T(1e-6)
u = randn(T, input_size)
Au = A*u
@test all(Au[1:padding[1][1], :] .== 0)
@test all(Au[end-padding[1][2]+1:end, :] .== 0)
@test all(Au[:, 1:padding[2][1]] .== 0)
@test all(Au[:, end-padding[2][2]+1:end] .== 0)
@test all(Au[padding[1][1]+1:end-padding[1][2], padding[2][1]+1:end-padding[2][2]] .== u)

# Adjoint test
rtol = T(1e-6)
u = randn(T, input_size)
v = randn(T, extended_size(input_size, padding))
@test adjoint_test(A; input=u, output=v, rtol=rtol)