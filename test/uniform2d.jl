using JLD, Test

ds = load(joinpath(@__DIR__, "..","datasets", "uniform_2d_10_pts.jld"))
V, (err, indices) = ds["data"], ds["size4_rank1"]
k = 1
n = 4

@test (find_low_rank_subset_checkall(V,n,k)) == (err, indices)
