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

"Returns (indices, elements)"
findmax(v::AbstractArray,k::AbstractUnitRange; opts...) = findpartialsort(v,k; opts..., rev=true)
"Returns (indices, elements)"
findmin(v::AbstractArray,k::AbstractUnitRange; opts...) = findpartialsort(v,k; opts...)

function findpartialsort(v,k; opts...)
    inds = partialsortperm(v,k;opts...)
    length(inds) == 1 ? (inds,v[inds]) : (inds,v[[inds...]])
end

"perform one step: find n closest points and compute principal vectors. Return (error, indices, SVD)"
function step(X, U, n :: Integer, k :: Integer)
    indices, S = find_n_closest_pts(X, U, n)
    err, svd = truncated_svd(hcat(S...), k)
    return err, indices, svd
end

"return n points from X closest to the span of U and their distances. Returns (indices, elements)"
function find_n_closest_pts(X, U, n)
    indices, _ = findmin([distance_squared(x, U) for x in X], 1:n)
    indices, X[indices]
end

"return square of distance from v to span of U, where columns of U are orthonormal"
distance_squared(v, U) = dot(v,v) - projection_squared(v, U)

"return square of length of projection of v onto the span of U, where columns of U are orthonormal"
projection_squared(v, U) = mapreduce(u->dot(v,u)^2 , +, eachcol(U))
