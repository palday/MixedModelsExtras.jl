using MixedModels
using MixedModels: dataset
using MixedModelsExtras
using Statistics
using Test

m = fit(MixedModel,
        @formula(reaction ~ 1 + days + (1+days|subj)),
        dataset(:sleepstudy))