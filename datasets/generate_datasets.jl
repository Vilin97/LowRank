"""
Generate datasets
"""
using Random

"generate points from uniform(0,1) × uniform(0,1)"
function uniform_2d(num_pts)
    [rand(2) for i in 1:num_pts]
end
