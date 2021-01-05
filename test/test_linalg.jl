using LinearAlgebra, Test, AbstractLinearOperators

# Loading custom-type module
include("./ModuleCustomType.jl")

# Random init
DT = Array{Float32, 4}
RT = Array{Float32, 4}
dsize = (2, 3, 4, 5)
rsize = (6, 7, 10, 2)
A = ModuleCustomType.CustomType{DT,RT}(dsize, rsize, randn(Float32,6*7,6))
B = ModuleCustomType.CustomType{DT,RT}(dsize, rsize, randn(Float32,6*7,6))
u = randn(Float32, dsize)
v = randn(Float32, rsize)

# Adjoint
@test dot(A*u, v) ≈ dot(u, adjoint(A)*v) rtol=1f-3

# +/-
C = A+B
@test C*u ≈ A*u+B*u rtol=1f-3
@test dot(C*u, v) ≈ dot(u, adjoint(C)*v) rtol=1f-3
C = A-B
@test C*u ≈ A*u-B*u rtol=1f-3
@test dot(C*u, v) ≈ dot(u, adjoint(C)*v) rtol=1f-3

# *
RT2 = Array{Float32, 4}
rsize2 = (7, 2, 5, 4)
v = randn(Float32, rsize2)
B = ModuleCustomType.CustomType{RT,RT2}(rsize, rsize2, randn(Float32,7*2,42))
C = B*A
@test C*u ≈ B*(A*u) rtol=1f-3
@test dot(C*u, v) ≈ dot(u, adjoint(C)*v) rtol=1f-3