"""
File defining Trajectory, SpatialTrajectory and related functions
"""

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
        return print(io, "error: $(traj.error)\nindices: $(traj.indices)")
    end

    "start with indices"
    function Trajectory(data_set, n, k, indices :: AbstractArray{I}) where I <: Integer
        @assert length(indices) == n
        error, svd = truncated_svd(hcat(data_set[indices]...), k)
        Trajectory(data_set, n, k, error, sort(indices), svd.U)
    end

    "pick the points closest to the span of vectors"
    function Trajectory(data_set, n, k, vectors :: Array{T, 2}) where T <: Number
        vectors = Matrix(qr(vectors).Q) # to orthonormalize principal_vectors
        indices, S = find_n_closest_pts(data_set, vectors, n)
        error, svd = truncated_svd(hcat(S...), k)
        Trajectory(data_set, n, k, error, sort(indices), svd.U)
    end

    "pick k points from the dataset at random and pick the rest n-k points to be the closest to the span of the k random points"
    function Trajectory(data_set, n, k)
        indices = sample(1:length(data_set), n)...)
        Trajectory(data_set, n, k, indices)
    end

    "perform one step: find n closest points and compute principal vectors. Return a new Trajectory object"
    function L2_step(trajectory)
        @unpack data_set, n, k, principal_vectors = trajectory
        indices, S = find_n_closest_pts(data_set, principal_vectors, n)
        error, svd = truncated_svd(hcat(S...), k)
        Trajectory(data_set, n, k, error, sort(indices), svd.U)
    end

"""
A struct to store all data needed to iterate, including a graph. principal_vectors have to be orthonormal
"""
struct SpatialTrajectory
    graph
    data_set
    n :: Int
    k :: Int
    λ :: Float64
    error :: Float64
    indices :: Array{Int,1}
    principal_vectors :: Array{Float64,2}
end

    function show(io :: IO, traj :: SpatialTrajectory)
        return print(io, "error: $(traj.error)\nindices: $(traj.indices)\nprincipal vectors: $(traj.principal_vectors)")
    end
