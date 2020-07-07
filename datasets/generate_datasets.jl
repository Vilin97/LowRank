" Generate datasets "
using JLD
using Random, Distributions

"generate points from uniform([0,1]ᵐ)"
function uniform_md(num_pts, m, seed)
    Random.seed!(seed)
    [rand(m) for i in 1:num_pts]
end

# seed = 123
# num_pts = 10
# m = 2
# ds = uniform_md(num_pts, m)
# save("datasets/uniform_2d_10_pts.jld", "data", ds, "size4_rank1", find_low_rank_subset_checkall(ds,4,1))
# @test ds == load("datasets/uniform_2d_10_pts.jld")["data"]

# A = UV plus N - n columns, where U is d×k, and V is k×n
d = 2
k = 1
n = 10
N = 30

A = zeros(d, N)
U = rand(d, k)
V = rand(k, n)
A[:, 1:n] = U*V
A[:, n+1:N] = rand(d, N-n)
# using Plots
# scatter(@view(A[1,:]), @view(A[2,:]))
