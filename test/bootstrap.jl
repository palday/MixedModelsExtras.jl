using MixedModels
using MixedModelsExtras
using Random
using Test

using MixedModels: likelihoodratiotest

sleepstudy = MixedModels.dataset(:sleepstudy)
fm0 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 | subj)), sleepstudy)
fm1 = fit(MixedModel, @formula(reaction ~ 1 + days + (1 + days | subj)), sleepstudy)
fmzc = fit(MixedModel, @formula(reaction ~ 1 + days + zerocorr(1 + days | subj)),
           sleepstudy)
