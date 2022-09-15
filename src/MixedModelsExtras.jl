module MixedModelsExtras

using LinearAlgebra
using MixedModels
using Statistics
using StatsBase
using StatsModels
using Tables

include("icc.jl")
export icc

include("r2.jl")
export r², r2, adjr², adjr2

include("vif.jl")
export termnames, gvif, vif

include("remef.jl")
export partial_fitted

end
