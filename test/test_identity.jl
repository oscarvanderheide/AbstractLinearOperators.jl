using AbstractLinearOperators, Test

# Linear operator
input_size = (2^7, 2^8)
T = ComplexF64
I = identity_operator(T; size=input_size)

# Identity test
rtol = real(T)(1e-6)
u = randn(T, input_size)
@test I*u ≈ u rtol=rtol

# Adjoint test
rtol = real(T)(1e-6)
@test adjoint_test(I; rtol=rtol)