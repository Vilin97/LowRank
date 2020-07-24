using LowRank, Test
# using Plots
# using JLD
using Random, Distributions, LinearAlgebra

N = 30
d = 2
n = 10
k = 1
err = 0.0
indices = 1:n

A = zeros(d, N)
U = rand(d, k)
V = rand(k, n)
A[:, 1:n] = U*V
A[:, n+1:N] = rand(d, N-n)
data_set = collect(eachcol(A))

# file = load(joinpath(@__DIR__, "..","datasets", "N_30_d_2.jld"))
# data_set, (err, indices) = file["data"], file["n_10_k_1"]
# n = 10
# k = 1

# @time @test find_low_rank_subset_checkall(data_set,n,k) == (err, indices)
@test find_low_rank_subset_sample_rep(data_set,n,k)[1] <= err + 0.0001
@test find_low_rank_subset_iterative(data_set, n, k)[1].error <= err + 0.0001

# PLotting
# p = scatter([Tuple(x) for x in V], label = "all points")
# scatter!(p, [Tuple(V[i]) for i in indices], label = "optimal points")
# plot!(p, [(0., 0.), -1 .* Tuple(truncated_svd(hcat(V[indices]...), 1).U[:,1])], label = "optimal line" )
