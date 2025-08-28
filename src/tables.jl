"""
    aictab(models::StatisticalModel...)

Generate a table of moe3l formulae and information criteria.

Required columns:
- `formula`: String representation of model formulae
- `DoF`: model degrees of freedom 
- `ΔAIC`: change in AIC from model with the lowest AIC value
- `ΔAICc`: change in AICc from model with the lowest BIC value 
- `ΔBIC`: change in BIC from model with the lowest BIC value

In the future additional required columns may be added without being considered breaking.
Additional non-required columns may be present, but their presence is not guaranteed until they become required.

No particular ordering of rows or columns is guaranteed.

Note that the minimum for the various information criteria may occur at different models.

The API guarantee is for a Tables.jl-compatible table, not for a specific return type.
"""
function aictab(models::StatisticalModel...)
    models = collect(models)
    aics = aic.(models)
    aiccs = aicc.(models)
    bics = bic.(models)
    loglik = loglikelihood.(models)
    DoF = dof.(models)

    ΔAIC = aics .- minimum(aics)
    ΔAICc = aiccs .- minimum(aiccs)
    ΔBIC = bics .- minimum(bics)

    return (; formula=string.(formula.(models)),
            DoF,
            Symbol("-2 loglikelihood") => -2 * loglik,
            ΔAIC,
            ΔAICc,
            ΔBIC)
end
