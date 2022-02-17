using MixedModels
using MixedModels: dataset
using MixedModelsExtras
using Test

@testset "LMM" begin
    fm0 = fit(MixedModel, @formula(reaction ~ 0 + days + (1|subj)), dataset(:sleepstudy))
    fm1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj)), dataset(:sleepstudy))
    fm2 = fit(MixedModel, @formula(reaction ~ 1 + days * days^2 + (1|subj)), dataset(:sleepstudy))

    ae_int = ArgumentError("VIF only defined for models with an intercept term")
    ae_nterm = ArgumentError("VIF not meaningful for models with only one non-intercept term")
    @test_throws ae_int vif(fm0)
    @test_throws ae_int vifnames(fm0)

    @test_throws ae_nterm vif(fm1)
    @test_throws ae_nterm vifnames(fm1)

    @test vifnames(fm2) == coefnames(fm2)[2:end]
    mm = @view cor(modelmatrix(fm2))[2:end, 2:end]
    # this is a slightly different way to do the same
    # computation. I've verified that doing it this way
    # in R gives the same answers as car::vif
    # note that computing the same model from scratch in R
    # gives different VIFs ([70.41934, 439.10096, 184.00107]),
    # but that model is a slightly different fit than the Julia
    # one and that has knock-on effects
    @test isapprox(vif(fm2), diag(inv(mm)))
end
