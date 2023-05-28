using DataFrames
using LinearAlgebra
using MixedModels
using MixedModelsExtras
using Test

using MixedModels: dataset
progress = false

m1 = fit(MixedModel,
         @formula(rt_trunc ~ 1 + spkr * prec * load +
                             (1 + spkr + prec + load | subj) +
                             (1 + spkr | item)),
         dataset(:kb07); progress)

st = shrinkagetables(m1)
for p in 1:3, grp in [:subj, :item]
    sn = DataFrame(shrinkagenorm(m1; p)[grp])
    sts = DataFrame(st[grp])
    cols = names(sts, !in(["subj", "item"]))
    sts = transform(sts,
                    cols => ByRow((x...) -> norm(x, p)) => :shrinkage)
    @test all(isapprox.(sts.shrinkage, sn.shrinkage; atol=0.005))
end
