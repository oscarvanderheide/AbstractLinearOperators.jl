#: Utils

export size_vec, deltype, dtype, reltype, rtype


size_vec(A::AbstractLinearOperator) = (prod(range_size(A)), prod(domain_size(A)))
eltype(A::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = T
deltype(A::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = eltype(DT)
dtype(A::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = DT
reltype(A::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = eltype(RT)
rtype(A::AbstractLinearOperator{T,DT,RT}) where {T,DT,RT} = RT
info(A::AbstractLinearOperator) = print("Linear operator: domain ≅ ", deltype(A), "^", domain_size(A), " → range ≅ ", reltype(A), "^", range_size(A))