using Aqua
using LinearAlgebra
using MixedModelsExtras
using Test

@testset "Aqua" begin
    # it's not piracy for StatsAPI.r2(::MixedModel), it's privateering!
    Aqua.test_all(MixedModelsExtras; ambiguities=false, piracy=false)
end

@testset "ICC" begin
    include("icc.jl")
end

@testset "r2" begin
    include("r2.jl")
end

@testset "VIF" begin
    include("vif.jl")
end

@testset "remef" begin
    include("remef.jl")
end

@testset "shrinkage" begin
    include("shrinkage.jl")
end
