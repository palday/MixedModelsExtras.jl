using MixedModels
using MixedModels: dataset
using MixedModelsExtras
using Statistics
using Test

function _adjust(r2, model)
    n = nobs(model)
    p = dof(model)
    return 1 - (1 - r2) * (n - 1) / (n - p)
end

progress = false

@testset "LMM" begin
    model = fit(MixedModel, @formula(reaction ~ 1 + (1|subj)),
                dataset(:sleepstudy); progress)

    @test r2(model;conditional=true) ≈ cor(response(model), fitted(model))^2
    @test r2(model;conditional=false) ≈ cor(response(model), model.X * model.β)^2

    @test adjr2(model;conditional=true) ≈ _adjust(r2(model; conditional=true), model)
    @test adjr2(model;conditional=false) ≈ _adjust(r2(model; conditional=false), model)
end
