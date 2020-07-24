" Generate datasets "
using JLD
using Random, Distributions, LinearAlgebra

N = 50
d = 5
n = 10
k = 2

# zero error
A = zeros(d, N)
U = rand(d, k)
V = rand(k, n)
A[:, 1:n] = U*V
A[:, n+1:N] = rand(d, N-n)
save("datasets/N_$(N)_d_$(d).jld", "data", collect(eachcol(A)), "n_$(n)_k_$(k)", (0.0, collect(1:n)))

# non-zero error
A = zeros(d, N)
U = rand(Uniform(-1.0, 1.0), d-1, k)
V = rand(Uniform(-1.0, 1.0), k, n)
A[1:d-1, 1:n] = U*V
A[d,1:n] = rand(Uniform(-1.0, 1.0), 1, n)
A[:, n+1:N] = rand(Uniform(-1.0, 1.0), d , N-n)
err, _ = LowRank.truncated_svd(A[:,1:n], k)
save("datasets/N_$(N)_d_$(d)_nonZeroError.jld", "data", collect(eachcol(A)), "n_$(n)_k_$(k)", (err, collect(1:n)))
