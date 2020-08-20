" Generate datasets "
# using JLD
using Random, Distributions

# N = 50
# d = 5
# n = 10
# k = 2

# zero error
# A = zeros(d, N)
# U = rand(d, k)
# V = rand(k, n)
# A[:, 1:n] = U*V
# A[:, n+1:N] = rand(d, N-n)
# save("datasets/N_$(N)_d_$(d).jld", "data", collect(eachcol(A)), "n_$(n)_k_$(k)", (0.0, collect(1:n)))

"""
generate N×d matrix A where the upper left p×n submatrix of A is of rank k, the rest Uniform[0,1] noise
"""
function matrix_with_subspace_noise(N, d, n, k, p)
    @assert p <= d
    @assert k < p
    A = rand(Uniform(-1.0, 1.0), d, N)
    U = rand(Uniform(-1.0, 1.0), p, k)
    V = rand(Uniform(-1.0, 1.0), k, n)
    A[1:p, 1:n] = U*V
    return A
end
# save("datasets/N_$(N)_d_$(d)_nonZeroError.jld", "data", collect(eachcol(A)), "n_$(n)_k_$(k)", (err, collect(1:n)))

"""
generate N×d matrix (A + M) where the the left d×n submatrix of A is of rank k,
the rest Uniform[0,1] noise, and M being a noise matrix with entries in Normal(0, std)
"""
function matrix_with_normal_noise(N, d, n, k, std)
    A = rand(Uniform(-1.0, 1.0), d, N)
    U = rand(Uniform(-1.0, 1.0), d, k)
    V = rand(Uniform(-1.0, 1.0), k, n)
    A[:, 1:n] = U*V
    noise = rand(Normal(0, std), d, N)
    return A + noise
end
