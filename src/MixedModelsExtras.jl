module MixedModelsExtras

export icc
export r², r2, adjr², adjr2
export termnames, gvif, vif

using LinearAlgebra
using MixedModels
using Statistics
using StatsBase
using StatsModels
using Tables

include("icc.jl")
include("r2.jl")
include("vif.jl")

end
