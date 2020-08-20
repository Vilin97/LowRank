"""
File with helper functions
"""

"return U, S, Vt truncated to rank k and the error ||M - M_k||^2"
function truncated_svd(M, k)
    @assert k <= min(size(M)...)
    F = svd(M)
    error = sum(F.S[k+1:end].^2)
    error, SVD(F.U[:,1:k], F.S[1:k], F.Vt[1:k,:])
end

"Returns (indices, elements)"
findmax(v::AbstractArray,k::AbstractUnitRange; opts...) = findpartialsort(v,k; opts..., rev=true)
"Returns (indices, elements)"
findmin(v::AbstractArray,k::AbstractUnitRange; opts...) = findpartialsort(v,k; opts...)

function findpartialsort(v,k; opts...)
    inds = partialsortperm(v,k;opts...)
    length(inds) == 1 ? (inds,v[inds]) : (inds,v[[inds...]])
end

"find n points from data_set closest to the span of U. Returns (indices, elements)"
function find_n_closest_pts(data_set, U, n)
    indices, _ = findmin([distance_squared(x, U) for x in data_set], 1:n)
    indices, data_set[indices]
end

"return square of distance from v to span of U, where columns of U are orthonormal"
distance_squared(v, U) = dot(v,v) - projection_squared(v, U)

"return square of length of projection of v onto the span of U, where columns of U are orthonormal"
projection_squared(v, U) = mapreduce(u->dot(v,u)^2 , +, eachcol(U))

"run method on data_set num_times times. Return errors, indices, and time took for each run"
function run_method(method, data_set, num_times)
    errors = []
    indicess = []
    times = []
    for i in 1:num_times
        t = @elapsed error, indices = method(data_set)
        push!(errors, error)
        push!(indicess, sort(indices))
        push!(times, t)
    end
    return errors, indicess, times
end
