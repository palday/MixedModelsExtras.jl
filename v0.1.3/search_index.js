var documenterSearchIndex = {"docs":
[{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"CurrentModule = MixedModelsExtras\nDocTestSetup = quote\n    using MixedModelsExtras\nend\nDocTestFilters = [r\"([a-z]*) => \\1\", r\"getfield\\(.*##[0-9]+#[0-9]+\"]","category":"page"},{"location":"api/#MixedModelsExtras.jl-API","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"","category":"section"},{"location":"api/#Coefficient-of-Determination","page":"MixedModelsExtras.jl API","title":"Coefficient of Determination","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"r2","category":"page"},{"location":"api/#StatsAPI.r2","page":"MixedModelsExtras.jl API","title":"StatsAPI.r2","text":"r2(model::StatisticalModel)\nr²(model::StatisticalModel)\n\nCoefficient of determination (R-squared).\n\nFor a linear model, the R² is defined as ESSTSS, with ESS the explained sum of squares and TSS the total sum of squares.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"adjr2","category":"page"},{"location":"api/#StatsAPI.adjr2","page":"MixedModelsExtras.jl API","title":"StatsAPI.adjr2","text":"adjr2(model::StatisticalModel)\nadjr²(model::StatisticalModel)\n\nAdjusted coefficient of determination (adjusted R-squared).\n\nFor linear models, the adjusted R² is defined as 1 - (1 - (1-R^2)(n-1)(n-p)), with R^2 the coefficient of determination, n the number of observations, and p the number of coefficients (including the intercept). This definition is generally known as the Wherry Formula I.\n\n\n\n\n\n","category":"function"},{"location":"api/#Intra-Class-Correlation-Coefficient","page":"MixedModelsExtras.jl API","title":"Intra-Class Correlation Coefficient","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"icc","category":"page"},{"location":"api/#MixedModelsExtras.icc","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.icc","text":"icc(model::MixedModel, [groups])\n\nCompute the intra-class correlation coefficient (ICC) for a mixed model.\n\nThe ICC is defined as the variance attributable to the groups divided by the total variance from all groups and the observation-level (residual) variance. In other words, the ICC can be interpreted as the proportion of the variance explainable by the grouping/nesting structure.\n\nA single group can be specified as a Symbol, e.g. :subj or a number of groups can be specified as an array: [:subj, :item]. If no groups are specified, then all grouping variables are used.\n\nnote: Note\nThe value returned here is sometimes called the \"adjusted ICC\" and does not take the variance of the fixed effects into account (the \"conditional ICC\").\n\nnote: Note\nThe result returned aggregates across groups. If you require the ICC for each group separately, then you must call icc separately for each group.\n\n\n\n\n\n","category":"function"},{"location":"api/#Variance-Inflation-Factor","page":"MixedModelsExtras.jl API","title":"Variance Inflation Factor","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"vif","category":"page"},{"location":"api/#MixedModelsExtras.vif","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.vif","text":"vif(m::RegressionModel)\n\nCompute the variance inflation factor (VIF).\n\nReturns a vector of inflation factors computed for each coefficient except for the intercept. In other words, the corresponding coefficients are coefnames(m)[2:end].\n\nThe variance inflation factor (VIF) measures the increase in the variance of a parameter's estimate in a model with multiple parameters relative to the variance of a parameter's estimate in a model containing only that parameter.\n\nSee also coefnames, gvif.\n\nwarning: Warning\nThis method will fail if there is (numerically) perfect multicollinearity, i.e. rank deficiency (in the fixed effects). In that case though, the VIF isn't particularly informative anyway.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"termnames","category":"page"},{"location":"api/#MixedModelsExtras.termnames","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.termnames","text":"termnames(model)\n\nReturn the names associated with (fixed effects) terms in a model.\n\nFor models with only continuous predictors, this is the same as coefnames(model). For models with categorical predictors, the returned names reflect the categorical predictor and not the coefficients resulting from the choice of contrast coding.\n\n\n\n\n\n","category":"function"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"gvif","category":"page"},{"location":"api/#MixedModelsExtras.gvif","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.gvif","text":"gvif(m::RegressionModel; scale=false)\n\nCompute the generalized variance inflation factor (GVIF).\n\nIf scale=true, then each GVIF is scaled by the degrees of freedom for (number of coefficients associated with) the predictor: GVIF^(1  (2*df))\n\nReturns a vector of inflation factors computed for each term except for the intercept. In other words, the corresponding coefficients are termnames(m)[2:end].\n\nThe generalized variance inflation factor (VIF) measures the increase in the variance of a (group of) parameter's estimate in a model with multiple parameters relative to the variance of a parameter's estimate in a model containing only that parameter. For continuous, numerical predictors, the GVIF is the same as the VIF, but for categorical predictors, the GVIF provides a single number for the entire group of contrast-coded coefficients associated with a categorical predictor.\n\nSee also termnames, vif.\n\nwarning: Warning\nThis method will fail if there is (numerically) perfect multicollinearity, i.e. rank deficiency (in the fixed effects). In that case though, the VIF isn't particularly informative anyway.\n\nReferences\n\nFox, J., & Monette, G. (1992). Generalized Collinearity Diagnostics. Journal of the American Statistical Association, 87(417), 178. doi:10.2307/2290467\n\n\n\n\n\n","category":"function"},{"location":"#MixedModelsExtras.jl-Documentation","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"","category":"section"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"CurrentModule = MixedModelsExtras","category":"page"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"MixedModelsExtras.jl is a Julia package providing extra capabilities for models fit with MixedModels.jl.","category":"page"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"Pages = [\n        \"api.md\",\n]\nDepth = 1","category":"page"}]
}
