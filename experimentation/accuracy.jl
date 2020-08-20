using LowRank

N = 500
n = 10
d = 10
k = 4
std = 0.01

num_times = 10
A = LowRank.matrix_with_normal_noise(N, d, n, k, std)
errors, indicess, times = LowRank.run_method(A -> FastLowRank(A, n, k), A, num_times)
baseline_error, _ = LowRank.truncated_svd(A[:,1:n], k)
baseline_indices = Set(1:n)
mean_error = sum(errors)/num_times
mean_indices_correctness = sum([Set(inds) == baseline_indices for inds in indicess])/num_times

inds = indicess[2]
data_set = collect(eachcol(A))
err, F = LowRank.truncated_svd(A[:,inds], k)
principal_vectors = F.U
traj = LowRank.Trajectory(data_set, n, k, err, inds, principal_vectors)






initial_indices = []; step = LowRank.L2_step; num_trajectories = 10; num_iterations = 50; verbose = true

current_trajectories = Vector{LowRank.Trajectory}(undef, num_trajectories)
# all_trajectories = Vector{Vector{Trajectory}}(undef, num_iterations)

# initialize
for t in 1:length(initial_indices)
    current_trajectories[t] = LowRank.Trajectory(data_set, n, k, initial_indices[t])
end
for t in length(initial_indices)+1:num_trajectories
    current_trajectories[t] = LowRank.Trajectory(data_set, n, k)
end
# all_trajectories[1] = current_trajectories

# do iterations
is_converged = [false for i in 1:num_trajectories]
global best_converged_error = Inf
global best_error = Inf
for i in 1:num_iterations
    for (ind, traj) in enumerate(current_trajectories)
        is_converged[ind] && continue
        new_traj = step(traj)
        global best_error = min(best_error, new_traj.error)
        if Set(traj.indices) == Set(new_traj.indices)
            is_converged[ind] = true
            global best_converged_error = min(best_converged_error, new_traj.error)
        end
        current_trajectories[ind] = new_traj
    end
    # all_trajectories[i+1] = current_trajectories
    if is_converged == [true for i in 1:num_trajectories]
        verbose && println("converged in $i steps")
        break
    elseif i == num_iterations
        verbose && println("did not converge in $i steps")
    end
end

errors = [t.error for t in current_trajectories]
is_converged
best_error
best_converged_error

traj = current_trajectories[9]
traj


# p = 4
# A = LowRank.matrix_with_subspace_noise(N, d, n, k, p)
