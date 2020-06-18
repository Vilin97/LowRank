"""
Module for identifying low-rank subsets and subgraphs
"""
module LowRank
using Combinatorics, LinearAlgebra

include("low_rank_subset.jl")

export find_low_rank_subset_checkall

end # module
