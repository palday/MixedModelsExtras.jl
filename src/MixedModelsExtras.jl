module MixedModelsExtras

export icc
export cooksdistance, influence, leverage
export r², r2, adjr², adjr2

using MixedModels
using Statistics
using StatsBase

include("icc.jl")
include("influence.jl")
include("leverage.jl")
include("cooksdistance.jl")
include("r2.jl")

end
