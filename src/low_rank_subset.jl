"""
Return (error, indices) for the subset of size n, whose k-rank approximation has lowest error, as measured by the 2-norm.
Checks all subsets of size n. Has comlpexity Ω(V^n).
"""
function find_low_rank_subset_checkall(data_set,n,k)
    f(S) = let M = hcat([data_set[i] for i in S]...);
        truncated_svd(M, k)[1], S end
    minimum(f , powerset(1:length(data_set), n, n) )
end

"""
Return (error, indices) for the subset of size n, whose k-rank approximation with vectors from data set has lowest error, as measured by the 2-norm.
Checks all subsets of size k. Has comlpexity Ω(V^k).
"""
function find_low_rank_subset_sample_rep(data_set,n,k)
    f(E) = let M = hcat([data_set[i] for i in E]...);
    M_times_pinv = M * pinv(M);
    distances_squared = [let d = v - M_times_pinv*v; dot(d,d) end for v in data_set]
    indices, distances = findmin(distances_squared, 1:n);
    sum(distances), indices
    end
    minimum(f, powerset(1:length(data_set), k, k))
end

struct Trajectory
    error :: Float64
    indices :: Array{Int,1}
    principal_components :: Array{Float64,2}
end

"""
Return an array of trajectories for each iteration step
"""
function find_low_rank_subset_iterative(data_set, n, k, num_trajectories = 5, num_iterations = 5, convergence_threshold = 0.01)
    trajectories = Array{Trajectory,1}(undef, num_trajectories)
    all_trajectories = Array{Trajectory,1}[]
    sizehint!(all_trajectories, num_iterations)
    for t in 1:num_trajectories
        U = hcat(sample(data_set, k)...)
        error, indices, svd = step(data_set, U, n, k)
        trajectories[t] = Trajectory(error, indices, svd.U)
    end
    push!(all_trajectories, trajectories)

    converged = false
    for i in 1:num_iterations
        for (t, traj) in enumerate(trajectories)
            trajectories[t] = Trajectory(step(data_set, traj.U, n, k))
            if abs(traj.error - trajectories[t].error) < convergence_threshold
                converged = true
                println("converged in $i steps. Error = $(trajectories[t].error)")
            end
        end
        push!(all_trajectories, trajectories)
        converged && break
    end
    return all_trajectories
end
