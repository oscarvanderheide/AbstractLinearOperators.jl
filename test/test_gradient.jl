T = Float64
input_size = 2^6
rtol = T(1e-6)
nb = 2

println("Testing gradient operator:")
for D = 1:3, batch = [false, true], device = [:cpu, :gpu]
    println("D=", D, "; batch=", batch, "; device=", device)

    # Gradient operator
    h = Tuple(abs.(randn(T, D)))
    ∇ = gradient_operator(h; batch=batch)

    # Adjoint test
    size_in = batch ? (input_size*ones(Integer, D)..., 1, nb) : Tuple(input_size*ones(Integer, D))
    size_out = batch ? ((input_size-1)*ones(Integer, D)..., D, nb) : ((input_size-1)*ones(Integer, D)..., D)
    if device == :cpu
        u = randn(T, size_in)
        v = randn(T, size_out)
    else
        u = CUDA.randn(T, size_in)
        v = CUDA.randn(T, size_out)
    end
    @test adjoint_test(∇; input=u, output=v, rtol=rtol)

    # Matrix coherency test
    if device == :cpu
        ∇mat = to_full_matrix(∇)
        if batch
            @test ∇mat*reshape(u, :, nb) ≈ reshape(∇*u, :, nb) rtol=rtol
        else
            @test ∇mat*vec(u) ≈ vec(∇*u) rtol=rtol
        end
    end

end
println()