__precompile__()

"""
Module for identifying low-rank subsets and subgraphs
"""
module LowRank
using Combinatorics, LinearAlgebra, StatsBase

import Base.Threads
@static if VERSION < v"1.3"
  seed_multiplier() = Threads.threadid()
else
  seed_multiplier() = 1
end

include("utils.jl")
include("low_rank_subset.jl")

export find_low_rank_subset_checkall, find_low_rank_subset_sample_rep, find_low_rank_subset_iterative

end # module
