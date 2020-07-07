############ Helper functions ################
"return U, S, Vt truncated to rank k and the error ||M - M_k||^2"
function truncated_svd(M, k)
    @assert k <= min(size(M)...)
    F = svd(M)
    error = sum(F.S[k+1:end].^2)
    error, SVD(F.U[:,1:k], F.S[1:k], F.Vt[1:k,:])
end

"multiply the three matrices in svd object"
combine_svd(svd_object) = svd_object.U*Diagonal(svd_object.S)*svd_object.Vt

"get the 2-norm error"
function error_2_norm(svd_object, original_matrix)
    norm(original_matrix - combine_svd(svd_object))
end

findmax(v::AbstractArray,k::AbstractUnitRange; opts...) = findpartialsort(v,k; opts..., rev=true)
findmin(v::AbstractArray,k::AbstractUnitRange; opts...) = findpartialsort(v,k; opts...)

function findpartialsort(v,k; opts...)
    inds = partialsortperm(v,k;opts...)
    length(inds) == 1 ? (inds,v[inds]) : (inds,v[[inds...]])
end

"sample n points from X by sampling k points randomly and selecting n-k points closest to the span on the first k"
function sample(X, n, k)
    U = hcat(sample(X, k))
    err, S = find_n_closest_pts(X, U, n)
end

"perform one step: compute principal vectors U and find n closest points. Return sum of errors and new set"
function step(X, U, n, k)
    err, S = find_n_closest_pts(X, U, n)
    svd = truncated_svd(hcat(S...), k)
    return error_2_norm(svd, )
end

"Return n points from X closest to the span of U and their distances"
find_n_closest_pts(X, U, n) = findmin([distance_squared(x, U) for x in X], 1:n)

"return square of length of projection of v onto the span of U, where columns of U are orthonormal"
projection_squared(v, U) = mapreduce(u->dot(v,u)^2 , +, eachcol(U))

"return square of distance from v to span of U, where columns of U are orthonormal"
distance_squared(v, U) = dot(v,v) - projection_squared(v, U)
