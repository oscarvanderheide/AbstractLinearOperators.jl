#: Utils

export size_vec, deltype, dtype, reltype, rtype


size_vec(A::AbstractLinearOperator) = (prod(range_size(A)), prod(domain_size(A)))
deltype(::AbstractLinearOperator{DT,RT}) where {DT,RT} = eltype(DT)
dtype(::AbstractLinearOperator{DT,RT}) where {DT,RT} = DT
reltype(::AbstractLinearOperator{DT,RT}) where {DT,RT} = eltype(RT)
rtype(::AbstractLinearOperator{DT,RT}) where {DT,RT} = RT
info(A::AbstractLinearOperator) = print("Linear operator: domain ≅ ", deltype(A), "^", domain_size(A), " → range ≅ ", reltype(A), "^", range_size(A))