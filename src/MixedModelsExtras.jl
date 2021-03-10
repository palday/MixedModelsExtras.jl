module MixedModelsExtras

export icc
export r², r2, adjr², adjr2

using MixedModels
using Statistics
using StatsBase

include("icc.jl")
include("r2.jl")

end
