"""
File defining Trajectory, SpatialTrajectory and related functions
"""

import Base.show
using Parameters
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

    function show(io :: IO, traj :: Trajectory)
        return print(io, "error: $(traj.error)\nindices: $(traj.indices)\nprincipal vectors: $(traj.principal_vectors)")
    end

    "pick the points closest to the span of vectors"
    function Trajectory(graph, data_set, n, k, nodes)
        vectors = Matrix(qr(data_set[nodes]).Q) # to orthonormalize principal_vectors
        indices, S = find_n_closest_pts(graph, data_set, nodes, n)
        error, svd = truncated_svd(hcat(S...), k)
        Trajectory(graph, data_set, n, k, error, indices, svd.U)
    end

    "pick k points from the dataset at random and pick the rest n-k points to be the closest to the span of the k random points"
    function Trajectory(graph, data_set, n, k)
        principal_vectors = hcat(sample(data_set, k)...)
        Trajectory(graph, data_set, n, k, principal_vectors)
    end
