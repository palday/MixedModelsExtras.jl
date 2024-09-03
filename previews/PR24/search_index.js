var documenterSearchIndex = {"docs":
[{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"CurrentModule = MixedModelsExtras\nDocTestSetup = quote\n    using MixedModelsExtras\nend\nDocTestFilters = [r\"([a-z]*) => \\1\", r\"getfield\\(.*##[0-9]+#[0-9]+\"]","category":"page"},{"location":"api/#MixedModelsExtras.jl-API","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"","category":"section"},{"location":"api/#Coefficient-of-Determination","page":"MixedModelsExtras.jl API","title":"Coefficient of Determination","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"r2","category":"page"},{"location":"api/#StatsAPI.r2","page":"MixedModelsExtras.jl API","title":"StatsAPI.r2","text":"r2(model::StatisticalModel)\nr²(model::StatisticalModel)\n\nCoefficient of determination (R-squared).\n\nFor a linear model, the R² is defined as ESSTSS, with ESS the explained sum of squares and TSS the total sum of squares.\n\n\n\n\n\nr2(model::StatisticalModel, variant::Symbol)\nr²(model::StatisticalModel, variant::Symbol)\n\nPseudo-coefficient of determination (pseudo R-squared).\n\nFor nonlinear models, one of several pseudo R² definitions must be chosen via variant. Supported variants are:\n\n:MacFadden (a.k.a. likelihood ratio index), defined as 1 - log (L)log (L_0);\n:CoxSnell, defined as 1 - (L_0L)^2n;\n:Nagelkerke, defined as (1 - (L_0L)^2n)(1 - L_0^2n).\n:devianceratio, defined as 1 - DD_0.\n\nIn the above formulas, L is the likelihood of the model, L_0 is the likelihood of the null model (the model with only an intercept), D is the deviance of the model (from the saturated model), D_0 is the deviance of the null model, n is the number of observations (given by nobs).\n\nThe Cox-Snell and the deviance ratio variants both match the classical definition of R² for linear models.\n\n\n\n\n\nr2(model::LinearMixedModel; conditional=false)\nr²(model::LinearMixedModel; conditional=false)\n\nCoefficient of determination (R-squared).\n\nR² is very non trivial for mixed models for a number of reasons and the measures here are particularly simplistic. For conditional=true, the Pearson correlation between the fitted and observed values is simply squared, in line with the relationship between the coefficient of determination and the Pearson correlation for classical OLS models. For conditional=false, new predicted values are generated based on only the fixed-effects estimates and the correlation between these predictions and the observed values is again squared.\n\nIn both cases, it is important to note that despite the usual naming convention (R² for the coefficient of determination and r for Pearson's correlation coefficient), these quantitaties are not defined in terms of the other. Instead, the correlation is a standardized measure of covariance and the coefficient of determination is measured relative to a null model. (The total sum of squares is in some sense the squared residual error of the null model.) Even generalizations of the coefficient of determination that define this value in terms of the likelihoods of an intercept-only model (i.e. the sample mean) and the likelihood of the fitted model fail to generalize to the mixed-model case: should \"intercept-only\" mean truly just the fixed effect (i.e. comparing the likelihoods of a classical OLS and a mixed-effects model)? or should it also include an random intercept for each grouping variable? What is the null model in this case, when determining whether each grouping variable contributes to overall fit? There is no single, clear, agreed-upon answer for these questions.  The bottom line is that there are many potential ways to define a coefficient of determination for linear mixed models (and even more for the generalzed case which combines all the problems of R² for GLM and for LMM) and none of them have all the properties of the coefficient of determination for classical OLS regression.\n\nThere are more advanced approximations (see e.g. Nakagawa and colleagues' 2013 2017 papers) but the fundamental philosophical issues above remain. In the future, additional approximations may be supported, but this is not a high priority.  Pull requests are welcome.\n\nFor more information, see the GLMM FAQ\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"adjr2","category":"page"},{"location":"api/#StatsAPI.adjr2","page":"MixedModelsExtras.jl API","title":"StatsAPI.adjr2","text":"adjr2(model::StatisticalModel)\nadjr²(model::StatisticalModel)\n\nAdjusted coefficient of determination (adjusted R-squared).\n\nFor linear models, the adjusted R² is defined as 1 - (1 - (1-R^2)(n-1)(n-p)), with R^2 the coefficient of determination, n the number of observations, and p the number of coefficients (including the intercept). This definition is generally known as the Wherry Formula I.\n\n\n\n\n\nadjr2(model::StatisticalModel, variant::Symbol)\nadjr²(model::StatisticalModel, variant::Symbol)\n\nAdjusted pseudo-coefficient of determination (adjusted pseudo R-squared). For nonlinear models, one of the several pseudo R² definitions must be chosen via variant. The only currently supported variants are :MacFadden, defined as 1 - (log (L) - k)log (L0) and :devianceratio, defined as 1 - (D(n-k))(D_0(n-1)). In these formulas, L is the likelihood of the model, L0 that of the null model (the model including only the intercept), D is the deviance of the model, D_0 is the deviance of the null model, n is the number of observations (given by nobs) and k is the number of consumed degrees of freedom of the model (as returned by dof).\n\n\n\n\n\nadjr2(model::LinearMixedModel; conditional=false)\nadjr²(model::LinearMixedModel; conditional=false)\n\nAdjusted coefficient of determination (adjusted R-squared).\n\nFor linear models, the adjusted R² is defined as (1 - (1-R^2)(n-1)(n-p)), with R^2 the coefficient of determination, n the number of observations, and p the number of coefficients (including the intercept). This definition is generally known as the Wherry Formula I.\n\nFor mixed-models, we use this same adjustment for the R² value computed with r2. conditional whether or not the random effects are taken into account when computing the quality of the model fit. They are nonetheless included in the adjustment (i.e. as number p of parameters).\n\nR² is not a clearly defined concept for mixed models, nor does it have all the properties typically expected of the coefficient of determination. See r2 for more information.\n\n\n\n\n\n","category":"function"},{"location":"api/#Intra-Class-Correlation-Coefficient","page":"MixedModelsExtras.jl API","title":"Intra-Class Correlation Coefficient","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"icc","category":"page"},{"location":"api/#MixedModelsExtras.icc","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.icc","text":"icc(model::MixedModel, [groups])\nicc(boot::MixedModelBootstrap, [family], [groups])\n\nCompute the intra-class correlation coefficient (ICC) for a mixed model.\n\nThe ICC is defined as the variance attributable to the groups divided by the total variance from all groups and the observation-level (residual) variance. In other words, the ICC can be interpreted as the proportion of the variance explainable by the grouping/nesting structure.\n\nA single group can be specified as a Symbol, e.g. :subj or a number of groups can be specified as an array: [:subj, :item]. If no groups are specified, then all grouping variables are used.\n\nWhen a MixedModelBootstrap is passed, a vector of ICC values for each bootstrap iteration is returned. Because MixedModelBootstrap does not store the associated model family for generalized linear mixed models, the family must be specified (e.g., Bernoulli(), Poisson()). A shortest coverage interval can be computed with MixedModels.shortestcovint.\n\nnote: Note\nThe value returned here is sometimes called the \"adjusted ICC\" and does not take the variance of the fixed effects into account (the \"conditional ICC\").\n\nnote: Note\nThe result returned aggregates across groups. If you require the ICC for each group separately, then you must call icc separately for each group.\n\n\n\n\n\n","category":"function"},{"location":"api/#Variance-Inflation-Factor","page":"MixedModelsExtras.jl API","title":"Variance Inflation Factor","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"vif","category":"page"},{"location":"api/#StatsAPI.vif","page":"MixedModelsExtras.jl API","title":"StatsAPI.vif","text":"vif(m::RegressionModel)\n\nCompute the variance inflation factor (VIF).\n\nThe VIF measures the increase in the variance of a parameter's estimate in a model with multiple parameters relative to the variance of a parameter's estimate in a model containing only that parameter.\n\nSee also gvif.\n\nwarning: Warning\nThis method will fail if there is (numerically) perfect multicollinearity, i.e. rank deficiency. In that case though, the VIF is not particularly informative anyway.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"termnames","category":"page"},{"location":"api/#StatsModels.termnames","page":"MixedModelsExtras.jl API","title":"StatsModels.termnames","text":"termnames(model::StatisticalModel)\n\nReturn the names of terms used in the formula of model.\n\nThis is a convenience method for termnames(formula(model)), which returns a two-tuple of termnames applied to the left and right hand sides of the formula.\n\nFor RegressionModels with only continuous predictors, this is the same as (responsename(model), coefnames(model)) and coefnames(formula(model)).\n\nFor models with categorical predictors, the returned names reflect the variable name and not the coefficients resulting from the choice of contrast coding.\n\nSee also coefnames.\n\n\n\n\n\ntermnames(t::FormulaTerm)\n\nReturn a two-tuple of termnames applied to the left and right hand sides of the formula.\n\nnote: Note\nUntil apply_schema has been called, literal 1 in formulae is interpreted as ConstantTerm(1) and will thus appear as \"1\" in the returned term names.\n\njulia> termnames(@formula(y ~ 1 + x * y + (1+x|g)))\n(\"y\", [\"1\", \"x\", \"y\", \"x & y\", \"(1 + x) | g\"])\n\nSimilarly, formulae with an implicit intercept will not have a \"1\" in their variable names, because the implicit intercept does not exist until apply_schema is called (and may not exist for certain model contexts).\n\njulia> termnames(@formula(y ~ x * y + (1+x|g)))\n(\"y\", [\"x\", \"y\", \"x & y\", \"(1 + x) | g\"])\n\n\n\n\n\ntermnames(term::AbstractTerm)\n\nReturn the name of the statistical variable associated with a term.\n\nReturn value is either a String, an iterable of Strings or the empty vector  if there is no associated variable (e.g. termnames(InterceptTerm{false}())).\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"gvif","category":"page"},{"location":"api/#StatsAPI.gvif","page":"MixedModelsExtras.jl API","title":"StatsAPI.gvif","text":"gvif(m::RegressionModel; scale=false)\n\nCompute the generalized variance inflation factor (GVIF).\n\nIf scale=true, then each GVIF is scaled by the degrees of freedom for (number of coefficients associated with) the predictor: GVIF^(1  (2*df)).\n\nThe GVIF measures the increase in the variance of a (group of) parameter's estimate in a model with multiple parameters relative to the variance of a parameter's estimate in a model containing only that parameter. For continuous, numerical predictors, the GVIF is the same as the VIF, but for categorical predictors, the GVIF provides a single number for the entire group of contrast-coded coefficients associated with a categorical predictor.\n\nSee also vif.\n\nReferences\n\nFox, J., & Monette, G. (1992). Generalized Collinearity Diagnostics. Journal of the American Statistical Association, 87(417), 178. doi:10.2307/2290467\n\n\n\n\n\n","category":"function"},{"location":"api/#\"Partial\"-Effects","page":"MixedModelsExtras.jl API","title":"\"Partial\" Effects","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"partial_fitted","category":"page"},{"location":"api/#MixedModelsExtras.partial_fitted","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.partial_fitted","text":"partial_fitted(model::MixedModel,\n               fe::AbstractVector{<:AbstractString},\n               re::Dict{Symbol}=Dict(fn => fe for fn in fnames(model));\n               mode=:include,\n               type=:linpred)\n\nCompute \"partial\" fitted values.\n\nPartial fitted values are useful for computing partial residuals. They are the fitted values obtained by setting selected model terms to zero, while preserving the other values at their original estimates.\n\nThe fixed effects coefficients to use (fe) are specified as vector of strings. For specifying no coefficients, use the empty string vector String[].\n\nThe random effects re are specified as a dictionary, with the grouping variables as keys and the vector of group-level coefficients specified as vectors. For example, Dict(:subj => [\"(Intercept)\"]) specifies that (1|subj) should be kept. The default is to match the specified fixed effects for all grouping variables, but note that this will fail when the fixed effects specification is incompatible with any grouping variable.\n\nThe keyword argument mode specifies whether the fixed and random effects specifications are treated as coefficients to :include or :exclude.\n\nFor GeneralizedLinearMixedModel, the keyword argument type specifies whether the predictions should be returned on the scale of linear predictor (:linpred) or on the response scale (:response).\n\nwarning: Warning\nPartial fitted values can be misleading for generalized linear mixed models on the response scale because of the nonlinear nature of the link function. For example, in logistic regression the partial fitted values are computed on the linear predictor scale, i.e. the log odds scale, and then transformed to the response scale, i.e. the probablitiy scale. However, a simple additive contribution on the log odds scale is not additive on the probability scale. More directly, it is impossible to decompose the effects of individual predictors into simple additive contributions on the original scale.\n\nnote: Note\nFor both the fixed and the random effects, the relevant entities are the coefficient names, not the original term names.\n\nwarning: Warning\nThe intercept is not automatically / implicitly included and must always be explicitly specified.\n\nwarning: Warning\nThis functionality has not been tested on and thus verified to work with models with rank-deficient fixed effects.\n\nThis functionality is similar to the remef package in R.\n\n\n\n\n\n","category":"function"},{"location":"api/#Shrinkage-Metrics","page":"MixedModelsExtras.jl API","title":"Shrinkage Metrics","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"shrinkagenorm","category":"page"},{"location":"api/#MixedModelsExtras.shrinkagenorm","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.shrinkagenorm","text":"shrinkagenorm(m::MixedModel{T},\n              θref::AbstractVector{T}=(isa(m, LinearMixedModel) ? 1e4 : 1) .*\n                                      m.optsum.initial;\n              uscale=false, p=2)\n\nReturns a NamedTuple of Tables.jl-tables norm of the change from OLS estimates (across all relevant coefficients) to BLUPs from the mixed model.\n\np corresponds to the L_p norms, i.e. p=2 is the Euclidean metric.\n\nEach entry in the named tuple corresponds to a single grouping term.\n\nwarning: Warning\nThis function is not thread safe because it temporarily mutates the passed model before restoring its original form.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"shrinkagetables","category":"page"},{"location":"api/#MixedModelsExtras.shrinkagetables","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.shrinkagetables","text":"shrinkagetables(m::MixedModel{T},\n                θref::AbstractVector{T}=(isa(m, LinearMixedModel) ? 1e4 : 1) .*\n                                        m.optsum.initial;\n                uscale=false) where {T}\n\nReturns a NamedTuple of Tables.jl-tables of the change from OLS estimates to BLUPs from the mixed model.\n\nEach entry in the named tuple corresponds to a single grouping term.\n\nwarning: Warning\nThis function is not thread safe because it temporarily mutates the passed model before restoring its original form.\n\n\n\n\n\n","category":"function"},{"location":"#MixedModelsExtras.jl-Documentation","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"","category":"section"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"CurrentModule = MixedModelsExtras","category":"page"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"MixedModelsExtras.jl is a Julia package providing extra capabilities for models fit with MixedModels.jl.","category":"page"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"Pages = [\n        \"api.md\",\n]\nDepth = 1","category":"page"}]
}
