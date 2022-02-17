const SymbolCollection = Union{Symbol,Tuple{Symbol,Vararg{Symbol}},AbstractVector{Symbol}}

function _group_var(vc::VarCorr, group::Symbol)
    return sum(abs2, getproperty(getproperty(vc.σρ, group), :σ))
end

function _group_var(vc::VarCorr, groups::SymbolCollection)
    return sum(_group_var.(Ref(vc), groups))
end

function _group_var(vc::VarCorr)
    groups = collect(propertynames(vc.σρ))
    return _group_var(vc, groups)
end

"""
    icc(model::MixedModel, [groups])

Compute the intra-class correlation coefficient (ICC) for a mixed model.

The ICC is defined as the variance attributable to the `groups` divided by
the total variance from all groups and the observation-level (residual) variance.
In other words, the ICC can be interpreted as the proportion of the variance explainable
by the grouping/nesting structure.

A single `group` can be specified as a Symbol, e.g. `:subj` or a number of groups
can be specified as an array: `[:subj, :item]`. If no `groups` are specified, then
all grouping variables are used.

!!! note
    The value returned here is sometimes called the "adjusted ICC" and does not take
    the variance of the fixed effects into account (the "conditional ICC").

!!! note
    The result returned aggregates across groups. If you require the ICC for
    each group separately, then you must call `icc` separately for each group.
"""
function icc(model::GeneralizedLinearMixedModel,
             groups::Union{Symbol,SymbolCollection})
    dispersion_parameter(model) &&
        throw(ArgumentError("GLMMs with dispersion parameters are not currently supported."))

    if model.resp.d isa Union{Binomial,Bernoulli}
        σ²res = π^2 / 3
    elseif model.resp.d isa Poisson
        σ²res = 1.0
    else
        throw(ArgumentError("Family $(typeof(model.resp.d)) currently unsupported, please file an issue."))
    end

    return _icc(VarCorr(model), groups, σ²res)
end

function icc(model::LinearMixedModel,
             groups::Union{Symbol,SymbolCollection})
    return _icc(VarCorr(model), groups, varest(model))
end

icc(model::MixedModel) = icc(model, fnames(model))

function _icc(vc::VarCorr, groups::Union{Symbol,SymbolCollection}, σ²res)
    σ²_α = _group_var(vc, groups) # random effect(s) in numerator
    σ² = σ²res + _group_var(vc)
    return σ²_α / σ²
end
