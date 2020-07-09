"""
Module for identifying low-rank subsets and subgraphs
"""
module LowRank
using Combinatorics, LinearAlgebra, StatsBase


include("utils.jl")
include("low_rank_subset.jl")

export find_low_rank_subset_checkall, find_low_rank_subset_sample_rep, find_low_rank_subset_iterative

end # module
