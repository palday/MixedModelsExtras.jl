
function bootstrap_lrtest(rng::AbstractRNG, n::Integer, m0::MixedModel, ms::MixedModel...; optsum_overrides=(;), progress=true)
    m0 = deepcopy(m0)
    ms = [deepcopy(m) for m in ms]
    
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

    return MixedModels.LikelihoodRatioTest(formulas,
                                           (dof=dofs, deviance=devs),
                                           (dofdiff=dofdiffs, deviancediff=devdiffs, pvalues=pvals),
                                           first(models) isa LinearMixedModel)
end
