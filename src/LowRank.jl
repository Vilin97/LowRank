__precompile__()

"""
Module for identifying low-rank subsets and subgraphs
"""
module LowRank
using Combinatorics, LinearAlgebra, StatsBase, Parameters
import Base.show
import Base.findmin, Base.findmax

include("generate_datasets.jl")
include("utils.jl")
include("trajectory.jl")
include("low_rank_subset.jl")

export find_low_rank_subset_checkall, find_low_rank_subset_sample_rep, FastLowRank

end # module

# TODO:
# allow for maximization of ∑ σᵢ² for i ≦ k instead of ∑ σᵢ² for i ≧ k+1
# allow for normalization in FastLowRank
# write a test that introduces Gaussian noise to each entry
# write spatial methods
# methods for objectives with normalization?
# write a method to find many low-rank subsets, as opposed to just one
