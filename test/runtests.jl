using LowRank, Test

@time begin
  @time @testset "Uniform 2d" begin include("uniform2d.jl") end
end
