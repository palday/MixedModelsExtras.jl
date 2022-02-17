using Documenter
using MixedModelsExtras

makedocs(; root=joinpath(dirname(pathof(MixedModelsExtras)), "..", "docs"),
         sitename="MixedModelsExtras",
         doctest=true,
         pages=["index.md"])

deploydocs(; repo="github.com/palday/MixedModelsExtras.jl.git", push_preview=true)
