using DataFrames
using MixedModels
using MixedModels: dataset
using MixedModelsExtras
using StableRNGs
using Statistics
using Test

progress = false

@testset "LMM" begin
    model = fit(MixedModel, @formula(reaction ~ 1 + (1 | subj)), dataset(:sleepstudy);
                progress)
    @test icc(model, :subj) == icc(model, [:subj]) == icc(model)
    @test icc(model, :subj) ≈ 0.37918288

    formula = @formula(rt_trunc ~ 1 + spkr * prec * load +
                                  (1 + spkr | subj) +
                                  (1 | item))
    model = fit(MixedModel, formula, dataset(:kb07); progress)
    @test icc(model, :subj) + icc(model, :item) ≈ icc(model)

    @testset "bootstrap" begin
        boot = parametricbootstrap(StableRNG(42), 100, model; hide_progress=!progress)
        iccboot_subj = icc(boot, :subj)
        iccboot_item = icc(boot, :item)
        @test iccboot_subj + iccboot_item ≈ icc(boot)

        ci_subj = shortestcovint(iccboot_subj)
        ci_item = shortestcovint(iccboot_item)
        @test first(ci_subj) < icc(model, :subj) < last(ci_subj)
        @test first(ci_item) < icc(model, :item) < last(ci_item)
    end
end

@testset "Binomial" begin
    cbpp = dataset(:cbpp)
    model = fit(MixedModel, @formula((incid / hsz) ~ 1 + (1 | herd)),
                cbpp, Binomial(); wts=float(cbpp.hsz), progress)
    @test icc(model, :herd) == icc(model, [:herd]) == icc(model)
    @test icc(model, :herd) ≈ 0.1668 atol = 0.0005
end

@testset "Bernoulli" begin
    contra = dataset(:contra)
    modelbern = fit(MixedModel, @formula(use ~ 1 + (1 | urban & dist)),
                    contra, Bernoulli(); fast=true, progress)
    # force treating as a Binomial model
    modelbin = fit(MixedModel, @formula(use ~ 1 + (1 | urban & dist)),
                   contra, Binomial(); fast=true, wts=ones(length(contra.dist)), progress)
    # Bernoullis are a special case of binomial, so make sure they give the same answer
    @test icc(modelbern, Symbol("urban & dist")) ≈ icc(modelbin, Symbol("urban & dist"))

    @testset "bootstrap" begin
        boot = parametricbootstrap(StableRNG(42), 100, modelbern; hide_progress=!progress)
        @test_throws ArgumentError icc(boot)
        iccboot = icc(boot, Bernoulli())
        ci = shortestcovint(iccboot)
        @test first(ci) < icc(modelbern) < last(ci)
        @test iccboot ≈ icc(boot, Bernoulli(), Symbol("urban & dist"))
    end
end

@testset "Poisson" begin
    grouseticks = DataFrame(dataset(:grouseticks))
    grouseticks.ch = grouseticks.height .- mean(grouseticks.height)
    model = fit(MixedModel,
                @formula(ticks ~ 1 + year + ch + (1 | index) + (1 | brood) + (1 | location)),
                grouseticks, Poisson(); fast=true, progress)
    @test icc(model, :index) ≈ 0.13467352262090606652 atol = 0.0005
    @test icc(model, [:index, :brood]) ≈ 0.3878770599741494518 atol = 0.0005
    @test icc(model, [:index, :brood, :location]) ≈ 0.53293244949745322003 atol = 0.0005
    @test icc(model, [:index, :brood, :location]) == icc(model)
end
