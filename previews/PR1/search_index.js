var documenterSearchIndex = {"docs":
[{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"CurrentModule = MixedModelsExtras\nDocTestSetup = quote\n    using MixedModelsExtras\nend\nDocTestFilters = [r\"([a-z]*) => \\1\", r\"getfield\\(.*##[0-9]+#[0-9]+\"]","category":"page"},{"location":"api/#MixedModelsExtras.jl-API","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"","category":"section"},{"location":"api/#Intra-Class-Correlation-Coefficient","page":"MixedModelsExtras.jl API","title":"Intra-Class Correlation Coefficient","text":"","category":"section"},{"location":"api/","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.jl API","text":"icc","category":"page"},{"location":"api/#MixedModelsExtras.icc","page":"MixedModelsExtras.jl API","title":"MixedModelsExtras.icc","text":"icc(model::MixedModel, [groups])\n\nCompute the intra-class correlation coefficient (ICC) for a mixed model.\n\nThe ICC is defined as the variance attributable to the groups divided by the total variance from all groups and the observation-level (residual) variance. In other words, the ICC can be interpreted as the proportion of the variance explainable by the grouping/nesting structure.\n\nA single group can be specified as a Symbol, e.g. :subj or a number of groups can be specified as an array: [:subj, :item]. If no groups are specified, then all grouping variables are used.\n\nnote: Note\nThe value returned here is sometimes called the \"adjusted ICC\" and does not take the variance of the fixed effects into account (the \"conditional ICC\").\n\nnote: Note\nThe result returned aggregates across groups. If you require the ICC for each group separately, then you must call icc separately for each group.\n\n\n\n\n\n","category":"function"},{"location":"#MixedModelsExtras.jl-Documentation","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"","category":"section"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"CurrentModule = MixedModelsExtras","category":"page"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"MixedModelsExtras.jl is a Julia package providing extra capabilities for models fit in with MixedModels.jl.","category":"page"},{"location":"","page":"MixedModelsExtras.jl Documentation","title":"MixedModelsExtras.jl Documentation","text":"Pages = [\n        \"api.md\",\n]\nDepth = 1","category":"page"}]
}
