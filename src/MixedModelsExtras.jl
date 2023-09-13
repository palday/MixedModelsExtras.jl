module MixedModelsExtras

using LinearAlgebra
using MixedModels
using Statistics
using StatsBase
using StatsModels
using Tables

using StatsModels: termnames, vif, gvif
export termnames, gvif, vif

StatsModels.termnames(::RandomEffectsTerm) = String[]

include("icc.jl")
export icc

include("r2.jl")
export r², r2, adjr², adjr2

include("remef.jl")
export partial_fitted

include("shrinkage.jl")
export shrinkagenorm, shrinkagetables

end
