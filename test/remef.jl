using GLM: linkinv, Link
using MixedModels
using MixedModels: dataset
using MixedModelsExtras
using Test

progress = false

@testset "LMM" begin
    fm1 = fit(MixedModel,
              @formula(reaction ~ 1 + days + (1 + days | subj)),
              dataset(:sleepstudy);
              progress)

    @test all(==(0), partial_fitted(fm1, String[]; mode=:include))
    @test all(partial_fitted(fm1, String[]; mode=:exclude) .≈ fitted(fm1))
    @test all(partial_fitted(fm1, ["(Intercept)", "days"], Dict(:subj => String[]);
                             mode=:include) .≈ fm1.X * fm1.β)

    @test_throws(ArgumentError("""specified FE names not subset of ["(Intercept)", "days"]"""),
                 partial_fitted(fm1, ["(Intercept)", "Days"], Dict(:subj => []);
                                mode=:include))
    @test_throws(ArgumentError("""specified RE names for subj not subset of ["(Intercept)", "days"]"""),
                 partial_fitted(fm1, ["(Intercept)", "days"], Dict(:subj => ["Days"]);
                                mode=:include))

    re_only_pf = partial_fitted(fm1, String[], Dict(:subj => String["(Intercept)", "days"]);
                                mode=:include)
    re_only = fitted(fm1) - fm1.X * fm1.β
    @test all(re_only .≈ re_only_pf)
end

@testset "GLMM" begin
    contra = dataset(:contra)
    gm1 = fit(MixedModel, @formula(use ~ 1 + (1 | urban & dist)),
              contra, Bernoulli(); fast=true, progress)

    pf_all = partial_fitted(gm1, ["(Intercept)"]; mode=:include, type=:response)
    fitted_vals = fitted(gm1)
    @test all(pf_all .≈ fitted_vals)

    re_only_pf = partial_fitted(gm1, String[],
                                Dict(Symbol("urban & dist") => String["(Intercept)"]);
                                mode=:include, type=:linpred)
    full = modelmatrix(gm1) * fixef(gm1) + re_only_pf
    @test all(linkinv.(Link(gm1), full) .≈ fitted_vals)
end
