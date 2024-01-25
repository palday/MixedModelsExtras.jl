
# Intercept always has to be specified
"""
    partial_fitted(model::MixedModel,
                   fe::AbstractVector{<:AbstractString},
                   re::Dict{Symbol}=Dict(fn => fe for fn in fnames(model));
                   mode=:include,
                   type=:linpred)

Compute "partial" fitted values.

Partial fitted values are useful for computing partial residuals. They are the
fitted values obtained by setting selected model terms to zero, while preserving
the other values at their original estimates.

The fixed effects coefficients to use (`fe`) are specified as vector of strings.
For specifying no coefficients, use the empty string vector `String[]`.

The random effects `re` are specified as a dictionary, with the grouping variables
as keys and the vector of group-level coefficients specified as vectors.
For example, `Dict(:subj => ["(Intercept)"])` specifies that `(1|subj)` should
be kept.
The default is to match the specified fixed effects for all grouping variables,
but note that this will fail when the fixed effects specification is incompatible
with any grouping variable.

The keyword argument `mode` specifies whether the fixed and random effects
specifications are treated as coefficients to `:include` or `:exclude`.

For `GeneralizedLinearMixedModel`, the keyword argument `type` specifies whether
the predictions should be returned on the scale of linear predictor (`:linpred`)
or on the response scale (`:response`).

!!! warning
    Partial fitted values can be misleading for generalized linear mixed models
    on the response scale because of the nonlinear nature of the link function.
    For example, in logistic regression the partial fitted values are computed
    on the linear predictor scale, i.e. the log odds scale, and then transformed
    to the response scale, i.e. the probablitiy scale. However, a simple additive
    contribution on the log odds scale is not additive on the probability scale.
    More directly, it is impossible to decompose the effects of individual predictors
    into simple additive contributions on the original scale.

!!! note
    For both the fixed and the random effects, the relevant entities are the
    coefficient names, not the original term names.

!!! warning
    The intercept is **not** automatically / implicitly included and must
    always be explicitly specified.

!!! warning
    This functionality has not been tested on and thus verified to work with
    models with rank-deficient fixed effects.

This functionality is similar to the [remef](https://github.com/hohenstein/remef) package in R.
"""
function partial_fitted(model::LinearMixedModel{T},
                        fe::AbstractVector{<:AbstractString},
                        re::Dict{Symbol}=Dict(fn => fe for fn in fnames(model));
                        mode=:include) where {T}
    return _partial_fitted(model, fe, re; mode)
end

function partial_fitted(model::GeneralizedLinearMixedModel{T},
                        fe::AbstractVector{<:AbstractString},
                        re::Dict{Symbol}=Dict(fn => fe for fn in fnames(model));
                        mode=:include, type=:linpred) where {T}
    type in (:linpred, :response) || throw(ArgumentError("Invalid value for type: $(type)"))
    y = _partial_fitted(model, fe, re; mode)
    return type == :linpred ? y : broadcast!(Base.Fix1(linkinv, Link(model)), y, y)
end

function _partial_fitted(model::MixedModel{T},
                         fe::AbstractVector{<:AbstractString},
                         re::Dict{Symbol}; mode) where {T}
    # @debug fe
    # @debug re
    issubset(fe, coefnames(model)) ||
        throw(ArgumentError("specified FE names not subset of $(coefnames(model))"))

    mode in [:include, :exclude] ||
        throw(ArgumentError("Invalid mode: $(mode)."))
    fe_idx = if isempty(fe)
        BitVector(false for c in fixefnames(model))
    else
        BitVector(c in fe for c in fixefnames(model))
    end
    # @debug fe_idx
    mode == :exclude && (fe_idx = .!fe_idx)
    # @debug fe_idx
    # XXX does this work properly for rank-deficient models?
    X = view(modelmatrix(model), :, fe_idx)
    vv = mul!(Vector{T}(undef, nobs(model)), X, view(fixef(model), fe_idx))

    for (rt, bb) in zip(model.reterms, ranef(model))
        group = Symbol(string(rt.trm))
        # @debug group
        # @debug group in keys(re)
        !isnothing(get(re, group, nothing)) || mode == :exclude || continue
        issubset(re[group], rt.cnames) ||
            throw(ArgumentError("specified RE names for $(group) not subset of $(rt.cnames)"))
        re_idx = if isempty(re[group])
            BitVector(false for c in rt.cnames)
        else
            BitVector(c in re[group] for c in rt.cnames)
        end
        mode == :exclude && (re_idx = .!re_idx)
        # @debug re_idx
        # nothing to do
        all(==(0), re_idx) && continue
        # @debug "not skipped"

        re_idx_reps = reduce(vcat, (re_idx for i in eachindex(rt.levels)))
        # @debug re_idx_reps
        # XXX no appropriate mul! method
        # mul!(vv, view(rt, :, re_idx_reps), view(bb, re_idx, :), one(T), one(T))
        mul!(vv, view(rt, :, re_idx_reps), vec(view(bb, re_idx, :)), one(T), one(T))

        # should re-write this as a loop to avoid allocating the intermediate allocation
        # @debug size(view(rt, :, re_idx_reps))
        # @debug size(view(bb, re_idx, :))
        # vv .+= view(rt, :, re_idx_reps) * vec(view(bb, re_idx, :))
    end

    return vv
end
