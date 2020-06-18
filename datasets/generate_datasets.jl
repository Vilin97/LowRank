" Generate datasets "
using JLD
using Random

"generate points from uniform([0,1]ᵐ)"
function uniform_md(num_pts, m, seed)
    Random.seed!(seed)
    [rand(m) for i in 1:num_pts]
end

seed = 123
num_pts = 10
m = 2
ds = uniform_md(num_pts, m)
save("datasets/uniform_2d_10_pts.jld", "data", ds, "size4_rank1", find_low_rank_subset_checkall(ds,4,1))
@test ds == load("datasets/uniform_2d_10_pts.jld")["data"]
