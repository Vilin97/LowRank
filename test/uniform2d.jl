using LowRank
# using Plots
using JLD, Test

# rank 1
file = load(joinpath(@__DIR__, "..","datasets", "uniform_2d_10_pts.jld"))
data_set, (err, indices) = file["data"], file["size4_rank1"]
k = 1
n = 4

# @test find_low_rank_subset_checkall(data_set,n,k)[1] <= err + 0.0001
@test find_low_rank_subset_sample_rep(data_set,n,k)[1] <= 2*err + .0001
@test find_low_rank_subset_iterative(data_set, n, k)[1].error <= err + 0.0001
# PLotting
# p = scatter([Tuple(x) for x in V], label = "all points")
# scatter!(p, [Tuple(V[i]) for i in indices], label = "optimal points")
# plot!(p, [(0., 0.), -1 .* Tuple(truncated_svd(hcat([V[i] for i in indices]...), 1).U[:,1])], label = "optimal line" )
# title!("2d, uniform")
