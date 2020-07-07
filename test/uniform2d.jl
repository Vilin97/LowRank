using LowRank
using Plots
using JLD, Test

# rank 1
ds = load(joinpath(@__DIR__, "..","datasets", "uniform_2d_10_pts.jld"))
V, (err, indices) = ds["data"], ds["size4_rank1"]
k = 1
n = 4

@test find_low_rank_subset_checkall(V,n,k) == (err, indices)
@test find_low_rank_subset_sample_rep(V,n,k)[1] <= 4*err

# PLotting
# p = scatter([Tuple(x) for x in V], label = "all points")
# scatter!(p, [Tuple(V[i]) for i in indices], label = "optimal points")
# plot!(p, [(0., 0.), -1 .* Tuple(truncated_svd(hcat([V[i] for i in indices]...), 1).U[:,1])], label = "optimal line" )
# title!("2d, uniform")
