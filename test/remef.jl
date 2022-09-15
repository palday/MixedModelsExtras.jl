using MixedModels
using MixedModels: dataset
using MixedModelsExtras
using Test

progress = false

fm1 = fit(MixedModel,
         @formula(reaction ~ 1 + days + (1 + days | subj)),
         dataset(:sleepstudy);
         progress)

@test all(==(0), partial_fitted(fm1, String[]; mode=:include))
@test all(partial_fitted(fm1, String[]; mode=:exclude) .≈ fitted(fm1))
@test all(partial_fitted(fm1, ["(Intercept)", "days"], Dict(:subj => String[]); mode=:include) .≈ fm1.X * fm1.β)

@test_throws(ArgumentError("""specified FE names not subset of ["(Intercept)", "days"]"""),
             partial_fitted(fm1, ["(Intercept)", "Days"], Dict(:subj => []); mode=:include))
@test_throws(ArgumentError("""ArgumentError: specified RE names for subj not subset of ["(Intercept)", "days"]"""),
             partial_fitted(fm1, ["(Intercept)", "days"], Dict(:subj => ["Days"]); mode=:include))

re_only_pf = partial_fitted(fm1, String[], Dict(:subj => String["(Intercept)", "days"]); mode=:include)
re_only = fitted(fm1) - fm1.X * fm1.β
@test all(re_only .≈ re_only_pf)
