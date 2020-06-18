using JLD, Test

V = load("datasets/uniform_2d_10_pts.jld")["data"]
k = 1
n = 3

@test find_low_rank_subset_checkall(V,n,k)[2] == [3, 5, 10]
