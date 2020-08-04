using LowRank, Test
# using JLD
# using Random, Distributions, LinearAlgebra

N = 50
d = 5
n = 10
k = 2
error = 0.0
indices = 1:n

A = zeros(d, N)
U = rand(d, k)
V = rand(k, n)
A[:, 1:n] = U*V
A[:, n+1:N] = rand(d, N-n)
data_set = collect(eachcol(A))

# file = load(joinpath(@__DIR__, "..","datasets", "N_50_d_5.jld"))
# data_set, (err, indices) = file["data"], file["n_10_k_2"]
# n = 10
# k = 2

# @time @test find_low_rank_subset_checkall(data_set,n,k) == (err, indices)
@test find_low_rank_subset_sample_rep(data_set,n,k)[1] <= 4*error + 0.0001
@test FastLowRank(data_set, n, k).error <= error + 0.0001
