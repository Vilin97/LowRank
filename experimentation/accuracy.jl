using LowRank
using Plots

function plot_changing_n(method, method_name, d, k, std, num_times, ns)
    mean_errors = []
    mean_indices_correctnesses = []
    mean_times = []
    baseline_errors = []
    for n in ns
        N = 10*n
        A = LowRank.matrix_with_normal_noise(N, d, n, k, std)
        errors, indicess, times = LowRank.run_method(B -> method(B, n, k), A, num_times)
        baseline_error, _ = LowRank.truncated_svd(A[:,1:n], k)
        baseline_indices = Set(1:n)
        mean_error = sum(errors)/num_times
        mean_indices_correctness = sum([length(intersect(inds,baseline_indices))/n for inds in indicess])/num_times
        mean_time = sum(times)/num_times
        push!(mean_errors, mean_error)
        push!(mean_indices_correctnesses, mean_indices_correctness)
        push!(mean_times, mean_time)
        push!(baseline_errors, baseline_error)
    end

    fontsize = 10
    texts = ["method: $method_name", "number of points N: 10n", "dimension d: $d", "target rank k: $k", "noise std: $std", "number of runs: $num_times"]
    y_locations = 0.9:-0.15:0
    annotations = [(0.2,y_locations[i], text(t, :left, fontsize)) for (i,t) in enumerate(texts)]
    p_text = plot([],[], ann=annotations, border =:none, label=:none)
    p_error = plot()
    plot!(p_error, ns, baseline_errors, label = "baseline errors")
    plot!(p_error, ns, mean_errors, label = "mean errors")
    p_indices = plot(ns, mean_indices_correctnesses, label = "mean indices correctness")
    ylims!(p_indices, (0,1))
    p_time = plot(ns, mean_times, label = "mean time")
    xaxis!(p_error, "size of target set, n")
    xaxis!(p_indices, "size of target set, n")
    xaxis!(p_time, "size of target set, n")
    p = plot(p_text,p_error, p_indices, p_time)
end

method_name = "Global FastLowRank"
method = FastLowRank
d = 10
k = 4
std = 0.1
num_times = 10
ns = 5:5:50
p = plot_changing_n(method, method_name, d, k, std, num_times, ns)
savefig(p, "GlobalFastLowRank.png")
method_name = "Local FastLowRank"
method = (B, n, k) -> FastLowRank(B, n, k, step = LowRank.L2_step_local)
p = plot_changing_n(method, method_name, d, k, std, num_times, ns)
savefig(p, "LocalFastLowRank.png")
