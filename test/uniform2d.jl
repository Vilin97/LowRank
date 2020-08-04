using LowRank

using JLD, Test

# rank 1
file = load(joinpath(@__DIR__, "..","datasets", "uniform_2d_10_pts.jld"))
data_set, (err, indices) = file["data"], file["size4_rank1"]
k = 1
n = 4

# @test find_low_rank_subset_checkall(data_set,n,k)[1] <= err + 0.0001
@test find_low_rank_subset_sample_rep(data_set,n,k)[1] <= 2*err + .0001
@test FastLowRank(data_set, n, k, verbose = true).error <= err + 0.0001

FastLowRank(data_set, n, k, verbose = true, num_trajectories = 100, convergence_threshold = 0.00000001)



# PLotting
# using Plots
# p = scatter([Tuple(x) for x in data_set], label = "all points",  series_annotations = text.(1:10, :bottom))
# scatter!(p, [Tuple(data_set[i]) for i in indices], label = "optimal points")
