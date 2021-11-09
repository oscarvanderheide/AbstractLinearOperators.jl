using LinearAlgebra, Test, AbstractLinearOperators

# Random init
input_size = (101, 200)
output_size = (101, 2, 100)
T = ComplexF32
A = reshape_operator(T, input_size, output_size)
u = randn(T, input_size)
v = randn(T, output_size)

# Adjoint
@test dot(A*u, v) ≈ dot(u, adjoint(A)*v) rtol=1f-6