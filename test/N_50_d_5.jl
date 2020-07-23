using LowRank
# using Plots
using JLD, Test

file = load(joinpath(@__DIR__, "..","datasets", "N_50_d_5.jld"))
data_set, (err, indices) = file["data"], file["n_10_k_2"]
n = 10
k = 2

# @time @test find_low_rank_subset_checkall(data_set,n,k) == (err, indices)
@time @test find_low_rank_subset_sample_rep(data_set,n,k)[1] <= 4*err + 0.0001
@time @test find_low_rank_subset_iterative(data_set, n, k)[1].error <= err + 0.0001
