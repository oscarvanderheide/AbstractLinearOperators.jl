using AbstractLinearOperators, CUDA, cuDNN, Test
CUDA.allowscalar(false)

T = Float64
input_size = 2^6
st_size = 3
nc_in = 2
nc_out = 3
nb = 4
rtol = T(1e-6)

for N = 1:3, device = [:cpu, :gpu]

    # Linear operator
    stencil = randn(T, st_size*ones(Integer, N)..., nc_in, nc_out)
    padding = Tuple([rand(0:st_size) for i = 1:2*N])
    C = convolution_operator(stencil; padding=padding, cdims_onthefly=false)

    # Adjoint test
    if device == :cpu
    u = randn(T, input_size*ones(Integer, N)..., nc_in, nb)
    output_size = size(C*u) # initialize
    v = randn(T, output_size)
    else
        u = CUDA.randn(T, input_size*ones(Integer, N)..., nc_in, nb)
        output_size = size(C*u) # initialize
        v = CUDA.randn(T, output_size)
    end
    @test adjoint_test(C; input=u, output=v, rtol=rtol)

    # Full-matrix coherence
    if device == :cpu
        Cmat = to_full_matrix(C)
        @test reshape(C*u, :, nb) ≈ Cmat*reshape(u, :, nb) rtol=rtol
    end

    # Stencil test
    if device == :cpu
        iseven(st_size) ? (st_size_ = st_size+1) : (st_size_ = st_size)
        u = zeros(T, st_size_*ones(Integer, N)..., 1, 1); u[(div(st_size_,2)+1)*ones(Integer, N)..., 1, 1] = 1
        stencil = randn(T, st_size_*ones(Integer, N)..., 1, nc_out)
        C = convolution_operator(stencil; padding=div(st_size_,2), flipped=false)
        Cu = C*u
        @test Cu ≈ reshape(C.stencil, st_size_*ones(Integer, N)..., nc_out, 1) rtol=rtol
    end

end