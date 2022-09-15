
# Intercept always has to be specified
"""
    partial_fitted(model::LinearMixedModel,
                        fe::AbstractVector{<:AbstractString},
                        re::Dict{Symbol}=Dict(fn => fe for fn in fnames(model));
                        mode=:include)

Compute "partial" fitted values.

Partial fitted values are useful for computing partial residuals.

This functionality is similar to the [remef](https://github.com/hohenstein/remef) package in R.
"""
function partial_fitted(model::LinearMixedModel{T},
                        fe::AbstractVector{<:AbstractString},
                        re::Dict{Symbol}=Dict(fn => fe for fn in fnames(model));
                        mode=:include) where {T}
    @debug fe
    @debug re
    issubset(fe, coefnames(model)) ||
        throw(ArgumentError("specified FE names not subset of $(coefnames(model))"))

    mode in [:include, :exclude] ||
        throw(ArgumentError("Invalid mode: $(mode)."))
    fe_idx = if isempty(fe)
        BitVector(false for c in fixefnames(model))
    else
        BitVector(c in fe for c in fixefnames(model))
    end
    @debug fe_idx
    mode == :exclude && (fe_idx = .!fe_idx)
    @debug fe_idx
    # XXX does this work properly for rank-deficient models?
    X = view(model.X, :, fe_idx)
    vv = mul!(Vector{T}(undef, nobs(model)), X, fixef(model)[fe_idx])

    for (rt, bb) in zip(model.reterms, ranef(model))
        group = Symbol(string(rt.trm))
        @debug group
        @debug group in keys(re)
        !isnothing(get(re, group, nothing)) || mode == :exclude || continue
        issubset(re[group], rt.cnames) ||
            throw(ArgumentError("specified RE names for $(group) not subset of $(rt.cnames)"))
        re_idx = if isempty(re[group])
            BitVector(false for c in rt.cnames)
        else
            BitVector(c in re[group] for c in rt.cnames)
        end
        mode == :exclude && (re_idx = .!re_idx)
        @debug re_idx
        # nothing to do
        all(==(0), re_idx) && continue
        @debug "not skipped"

        re_idx_reps = reduce(vcat, (re_idx for i in eachindex(rt.levels)))
        @debug re_idx_reps
        # XXX no appropriate mul! method
        # mul!(vv, view(rt, :, re_idx_reps), view(bb, re_idx, :), one(T), one(T))
        # should re-write this as a loop to avoid allocating the intermediate allocation
        # @debug size(view(rt, :, re_idx_reps))
        # @debug size(view(bb, re_idx, :))
        vv .+= view(rt, :, re_idx_reps) * vec(view(bb, re_idx, :))
    end

    return vv
end
