using Documenter
using MixedModelsExtras

makedocs(;
         repo=Remotes.GitHub("palday", "MixedModelsExtras.jl"),
         sitename="MixedModelsExtras",
         doctest=true,
         checkdocs=:exports,
         warnonly=[:cross_references],
         pages=["index.md", "api.md"])

deploydocs(; repo="github.com/palday/MixedModelsExtras.jl.git", push_preview=true)
