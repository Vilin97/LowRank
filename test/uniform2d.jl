using LowRank
using JLD, Test
using LinearAlgebra, Combinatorics

# rank 1
ds = load(joinpath(@__DIR__, "..","datasets", "uniform_2d_10_pts.jld"))
V, (err, indices) = ds["data"], ds["size4_rank1"]
k = 1
n = 4

@test find_low_rank_subset_checkall(V,n,k) == (err, indices)
@test find_low_rank_subset_sample_rep(V,n,k)[1] <= 4*err
