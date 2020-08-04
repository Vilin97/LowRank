using Parameters
import Base.findmin, Base.findmax, Base.show

"""
A struct to store all data needed to iterate. principal_vectors have to be orthonormal
"""
struct Trajectory
    data_set
    n :: Int
    k :: Int
    error :: Float64
    indices :: Array{Int,1}
    principal_vectors :: Array{Float64,2}
end

function show(io :: IO, traj :: Trajectory)
    return print(io, "error: $(traj.error)\nindices: $(traj.indices)\nprincipal vectors: $(traj.principal_vectors)")
end

"pick the points closest to the span of vectors"
function Trajectory(data_set, n, k, vectors)
    vectors = Matrix(qr(vectors).Q) # to orthonormalize principal_vectors
    indices, S = find_n_closest_pts(data_set, vectors, n)
    error, svd = truncated_svd(hcat(S...), k)
    Trajectory(data_set, n, k, error, indices, svd.U)
end

"pick k points from the dataset at random and pick the rest n-k points to be the closest to the span of the k random points"
function Trajectory(data_set, n, k)
    principal_vectors = hcat(sample(data_set, k)...)
    Trajectory(data_set, n, k, principal_vectors)
end

"perform one step: find n closest points and compute principal vectors. Return (error, indices, SVD)"
function L2_step(trajectory)
    @unpack data_set, n, k, principal_vectors = trajectory
    indices, S = find_n_closest_pts(data_set, principal_vectors, n)
    error, svd = truncated_svd(hcat(S...), k)
    Trajectory(data_set, n, k, error, indices, svd.U)
end

############ Helper functions ################
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
