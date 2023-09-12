module MixedModelsExtras

using LinearAlgebra
using MixedModels
using Random
using Statistics
using StatsBase
using StatsModels
using Tables

using MixedModels: replicate

include("icc.jl")
export icc

include("r2.jl")
export r², r2, adjr², adjr2

include("vif.jl")
export termnames, gvif, vif

include("remef.jl")
export partial_fitted

include("shrinkage.jl")
export shrinkagenorm, shrinkagetables

include("bootstrap.jl")
export bootstrap_lrt

end
