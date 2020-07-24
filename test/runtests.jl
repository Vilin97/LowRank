using LowRank, Test

begin
  @time @testset "10 pts, 2d" begin include("uniform2d.jl") end
  @time @testset "30 pts, 2d" begin include("N_30_d_2.jl") end
  @time @testset "50 pts, 5d" begin include("N_50_d_5.jl") end
  @time @testset "non-zero error" begin include("non_zero_error.jl") end
end
