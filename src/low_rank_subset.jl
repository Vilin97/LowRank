"""
Methods to find a low-rank subset
"""

"""
Return (error, indices) for the subset of size n, whose k-rank approximation has lowest error, as measured by the 2-norm.
Checks all subsets of size n. Has complexity Ω(|data_set|^n).
"""
function find_low_rank_subset_checkall(data_set,n,k)
    f(S) = let M = hcat([data_set[i] for i in S]...);
        truncated_svd(M, k)[1], S end
    minimum(f , powerset(1:length(data_set), n, n) )
end

find_low_rank_subset_checkall(A :: Array{T, 2}, n, k) where T <: Number = find_low_rank_subset_checkall(collect(eachcol(A)), n, k)

"""
Return (error, indices) for the subset of size n, whose k-rank approximation with vectors from data set has lowest error, as measured by the 2-norm.
Checks all subsets of size k. Has complexity Ω(|data_set|^k).
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

find_low_rank_subset_sample_rep(A :: Array{T, 2}, n, k) where T <: Number = find_low_rank_subset_sample_rep(collect(eachcol(A)), n, k)

"""
Return an array of current_trajectories for each iteration step.
"""
function FastLowRank(data_set, n, k; initial_indices = [], step = L2_step, num_trajectories = k*trunc(Int(length(data_set)/n)), num_iterations = 50, verbose = false)
    current_trajectories = Vector{Trajectory}(undef, num_trajectories)
    # all_trajectories = Vector{Vector{Trajectory}}(undef, num_iterations)

    # initialize
    for t in 1:length(initial_indices)
        current_trajectories[t] = LowRank.Trajectory(data_set, n, k, initial_indices[t])
    end
    for t in length(initial_indices)+1:num_trajectories
        current_trajectories[t] = LowRank.Trajectory(data_set, n, k)
    end

    # do iterations
    is_converged = [false for i in 1:num_trajectories]
    best_converged_error = Inf
    best_error = Inf
    for i in 1:num_iterations
        for (ind, traj) in enumerate(current_trajectories)
            is_converged[ind] && continue
            new_traj = step(traj)
            best_error = min(best_error, new_traj.error)
            if Set(traj.indices) == Set(new_traj.indices)
                is_converged[ind] = true
                best_converged_error = min(best_converged_error, new_traj.error)
            end
            current_trajectories[ind] = new_traj
        end

        # @show best_error, best_converged_error
        if best_converged_error - best_error < eps(best_error)
            verbose && println("converged in $i steps")
            break
        elseif i == num_iterations
            verbose && println("did not converge in $i steps")
        end
    end
    best_error, best_trajectory_index = findmin((x -> x.error).(current_trajectories))
    best_trajectory = current_trajectories[best_trajectory_index]
    verbose && println("error = $best_error")
    return best_trajectory.error, best_trajectory.indices
end

FastLowRank(A :: Array{T, 2}, n, k; kwargs...) where T <: Number = FastLowRank(collect(eachcol(A)), n, k; kwargs...)
