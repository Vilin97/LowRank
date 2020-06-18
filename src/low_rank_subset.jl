"""
Return indices for the subset of size n, whose k-rank approximation has lowest error, as measured by the 2-norm.
Checks all subsets of size n. Has comlpexity Ω(V^n).
"""
function find_low_rank_subset_checkall(V,n,k)
    minimum( S -> let M = hcat([V[i] for i in S]...); (error_2_norm(truncated_svd(M, k), M), S) end, powerset(1:length(V), n, n) )
end


############ Helper functions ################
"return U, S, Vt truncated to rank k"
function truncated_svd(M, k)
    @assert k <= min(size(M)...)
    F = svd(M)
    SVD(F.U[:,1:k], F.S[1:k], F.Vt[1:k,:])
end

"get the 2-norm error"
function error_2_norm(svd_object, original_matrix)
    norm(original_matrix - svd_object.U*Diagonal(svd_object.S)*svd_object.Vt)
end
