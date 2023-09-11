
function bootstrap_lrtest(rng::AbstractRNG, n::Integer, m0::MixedModel, m1::MixedModel; optsum_overrides=(;), progress=true)
    models = [m0, m1]
    dofs = dof.(models)
    formulas = string.(formula.(models))
    ord = sortperm(dofs)
    dofs = dofs[ord]
    formulas = formulas[ord]
    devs = deviance.(models)[ord]
    dofdiffs = diff(dofs)
    devdiffs = .-(diff(devs))
  
    m0 = deepcopy(m0)
    m1 = deepcopy(m1)
    for (key, val) in pairs(optsum_overrides)
        setfield!(m0.optsum, key, val)
        setfield!(m1.optsum, key, val)
    end
    nulldist = replicate(n; hide_progress=!progress) do
        simulate!(rng, m0)
        refit!(m0; progress=false)
        refit!(m1, response(m0))
        return deviance(m1) - deviance(m0)
    end
    pvals = map(devdiffs) do dev
        if dev > 0
            mean(>(dev), nulldist)
        else
            NaN
        end
    end

    return MixedModels.LikelihoodRatioTest(formulas,
                                           (dof=dofs, deviance=devs),
                                           (dofdiff=dofdiffs, deviancediff=devdiffs, pvalues=pvals),
                                           first(models) isa LinearMixedModel)
end
