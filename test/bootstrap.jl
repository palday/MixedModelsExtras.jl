using MixedModels
using MixedModelsExtras
using Random
using Test

using MixedModels: likelihoodratiotest

progress = false
sleepstudy = MixedModels.dataset(:sleepstudy)
fm0 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)),
          sleepstudy; progress)
fm1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)),
          sleepstudy; progress)
fmzc = fit(MixedModel, @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
           sleepstudy; progress)

lrt = likelihoodratiotest(fm0, fm1, fmzc)
boot = bootstrap_lrt(MersenneTwister(42), 1000, fm0, fm1, fmzc)
