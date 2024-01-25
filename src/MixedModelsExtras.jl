module MixedModelsExtras

using LinearAlgebra
using MixedModels
using Random
using Statistics
using StatsBase
using StatsModels
using Tables

using GLM: linkinv, Link
using MixedModels: replicate
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

include("bootstrap.jl")
export bootstrap_lrt

end
