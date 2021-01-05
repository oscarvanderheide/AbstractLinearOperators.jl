module ModuleCustomType

using AbstractLinearOperators
import AbstractLinearOperators: domain_size, range_size, matvecprod, matvecprod_adj

export CustomType

struct CustomType{DT,RT} <: AbstractLinearOperator{Float32,DT,RT}
    dsize::Tuple
    rsize::Tuple
    A::Array{Float32,2}
end

domain_size(A::CustomType) = A.dsize
range_size(A::CustomType) = A.rsize
matvecprod(A::CustomType{DT,RT}, u::DT) where {DT,RT} = reshape(A.A*reshape(u, A.dsize[1]*A.dsize[2], A.dsize[3]*A.dsize[4]), A.rsize)
matvecprod_adj(A::CustomType{DT,RT}, v::RT) where {DT,RT} = reshape(adjoint(A.A)*reshape(v, A.rsize[1]*A.rsize[2], A.rsize[3]*A.rsize[4]), A.dsize)

end