using LinearAlgebra, Test, AbstractLinearOperators

# Custom-type module
module ModuleCustomType
    using AbstractLinearOperators
    import AbstractLinearOperators: domain_size, range_size, matvecprod, matvecprod_adj

    export CustomType

    struct CustomType{ND,NR}<:AbstractLinearOperator{Float32,ND,NR}
        dsize::NTuple{ND,Int64}
        rsize::NTuple{NR,Int64}
        A::Array{Float32,2}
    end

    domain_size(A::CustomType) = A.dsize
    range_size(A::CustomType) = A.rsize
    matvecprod(A::CustomType{ND,NR}, u::AbstractArray{Float32,ND}) where {ND,NR} = reshape(A.A*reshape(u, A.dsize[1]*A.dsize[2], A.dsize[3]*A.dsize[4]), A.rsize)
    matvecprod_adj(A::CustomType{ND,NR}, v::AbstractArray{Float32,NR}) where {ND,NR} = reshape(adjoint(A.A)*reshape(v, A.rsize[1]*A.rsize[2], A.rsize[3]*A.rsize[4]), A.dsize)
end
using .ModuleCustomType

# Random init
dsize = (2, 3, 4, 5)
rsize = (6, 7, 10, 2)
A = CustomType{4,4}(dsize, rsize, randn(Float32,6*7,6))
B = CustomType{4,4}(dsize, rsize, randn(Float32,6*7,6))
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
B = CustomType(rsize, rsize2, randn(Float32,7*2,42))
C = B*A
@test C*u ≈ B*(A*u) rtol=1f-3
@test dot(C*u, v) ≈ dot(u, adjoint(C)*v) rtol=1f-3