using LowRank, Test
using Random, Distributions

N = 50
d = 5
n = 10
k = 2

# non-zero error
A = zeros(d, N)
U = rand(Uniform(-1.0, 1.0), d-1, k)
V = rand(Uniform(-1.0, 1.0), k, n)
A[1:d-1, 1:n] = U*V
A[d,1:n] = rand(Uniform(-1.0, 1.0), 1, n)
A[:, n+1:N] = rand(Uniform(-1.0, 1.0), d , N-n)
error, _ = LowRank.truncated_svd(A[:,1:n], k)
data_set = collect(eachcol(A))

@test find_low_rank_subset_sample_rep(data_set,n,k)[1] <= 4*error
@test FastLowRank(data_set, n, k)[1] <= error + 0.0001
