using AbstractLinearOperators, Test

# Linear operator
input_size = (2^7, 2^8)
T = ComplexF64
Id = identity_operator(T, 2; size=input_size)

# Identity test
rtol = real(T)(1e-6)
u = randn(T, input_size)
@test Id*u ≈ u rtol=rtol

# Adjoint test
rtol = real(T)(1e-6)
@test adjoint_test(Id; rtol=rtol)

# Full matrix coherence
Idmat = to_full_matrix(Id)
@test vec(Id*u) ≈ Idmat*vec(u) rtol=rtol