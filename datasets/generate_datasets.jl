" Generate datasets "
using JLD
using Random

"generate points from uniform(0,1) × uniform(0,1)"
function uniform_md(num_pts, m)
    [rand(m) for i in 1:num_pts]
end

num_pts = 10
m = 2
ds = uniform_md(num_pts, m)
save("datasets/uniform_2d_10_pts.jld", "data", ds)
ds == load("datasets/uniform_2d_10_pts.jld")["data"]
