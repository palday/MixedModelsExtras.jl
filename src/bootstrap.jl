"""
    bootstrap_lrtest(rng::AbstractRNG, n::Integer, m0::MixedModel, ms::MixedModel...;
                     optsum_overrides=(;), progress=true)

Bootstrapped likelihood ratio test applied to a set of nested models.

The first model is used to simulate `n` dataset replicates, where the ground truth is that
specified by the first model. Each of the other models is then refit to those
null data and the underlying distribution of deviance differences is then captured.
For final computation of the p-values, the observed deviance is differencce between the 
original models is compared against this null distribution.

!!! note
    The precision of the resulting p-value cannot exceed ``1/n``. 

!!! warn
    This method is **not** thread safe. For efficiency , the models are modified 
    during bootstrapping and the original fits are only restored at the end.

!!! note
    The nesting of the models is not checked.  It is incumbent on the user
    to check this. This differs from `StatsModels.lrtest` as nesting in
    mixed models, especially in the random effects specification, may be non obvious.

This functionality may be deprecated in the future in favor of `StatsModels.lrtest`.
"""
function bootstrap_lrtest(rng::AbstractRNG, n::Integer, m0::MixedModel, ms::MixedModel...;
                          optsum_overrides=(;), progress=true)
    y0 = response(m0)
    ys = [response(m) for m in ms]
    try
        models = [m0; ms...]
        dofs = dof.(models)
        formulas = string.(formula.(models))
        ord = sortperm(dofs)
        dofs = dofs[ord]
        formulas = formulas[ord]
        devs = deviance.(models)[ord]
        dofdiffs = diff(dofs)
        devdiffs = .-(diff(devs))

        for (key, val) in pairs(optsum_overrides)
            setfield!(m0.optsum, key, val)
            for m in ms
                setfield!(m.optsum, key, val)
            end
        end
        nulldist = replicate(n; hide_progress=!progress) do
            simulate!(rng, m0)
            refit!(m0; progress=false)
            for m in ms
                refit!(m, response(m0); progress=false)
            end
            return [deviance(m) for m in models]
        end
        nulldist = stack(nulldist; dims=1)
        nulldist = -1 .* diff(nulldist; dims=2)
        pvals = map(enumerate(devdiffs)) do (idx, dev)
            if dev > 0
                mean(>=(dev), view(nulldist, :, idx))
            else
                NaN
            end
        end
        # catch ex
        #     rethrow(ex)
    finally
        # restore the original fits
        if progress
            @info "Bootstrapping complete, cleaning up..."
        end
        refit!(m0, y0; progress=false)
        for (m, y) in zip(ms, ys)
            refit!(m, y; progress=false)
        end
    end
    return MixedModels.LikelihoodRatioTest(formulas,
                                           (dof=dofs, deviance=devs),
                                           (dofdiff=dofdiffs, deviancediff=devdiffs,
                                            pvalues=pvals),
                                           first(models) isa LinearMixedModel)
end
