using MixedModelsExtras
using Test

@testset "ICC" begin
    include("icc.jl")
end

@testset "r2" begin
    include("r2.jl")
end

@testset "VIF" begin
    include("vif.jl")
end
