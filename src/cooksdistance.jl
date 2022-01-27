"""
    cooksdistance(model)

Compute Cook's distance for each observation.
Cook's distance is defined as `` (e_i^2)/(p*MSE) * (h_i) / (1 - h_i)^2``
where ``e_i`` is the residual and ``h_i`` is the leverage for
observation ``i``, ``p`` is the number of parameters in the model and
``MSE`` is the mean squared error of the model.

Depending on the context, there are several possible thresholds.

If using absolute/invariant thresholds, then generally speaking points with
``D > 1.0`` will very probably be influential, points with ``D > 0.5`` are
often influential, and points with ``D < 0.5`` are probably not influential.
These absolute thresholds may not be the best guide; there are also thresholds
 based on the number of observations and the number of model parameters.

 Another possible treshold is the 50% quantile of the CDF of an ``F(p, n-p)``
distribution, i.e. with `dof(model)` and `dof_residual(model)` degrees of
freedom. This can be computed like so:
```jldoctest
julia> using Distributions, MixedModels
julia> m = fit(MixedModel, @formula(reaction ~ 1 + days + (1|subj)), MixedModels.dataset(:sleepstudy))
julia> quantile(FDist(dof(m), dof_residual(m)), 0.5)
0.8423977003344489
```

Other thresholds include ``4/n`` or ``4/(n-p-1)``, which can be very strict
for models with a large number of observations.

Note that this formula may be too naive for mixed models, as this measure was developed in the context of classical OLS models. As for all things with mixed models, the stratification of the error leads to some subtleties in interpretation. See also [`StatsBase.r2`](@ref).
"""
function StatsBase.cooksdistance(model::LinearMixedModel)
    # Given that this only depends on things from the StatsBase API,
    # we should probably contribute this back upstream
    p = dof(model)

    sqrerr = residuals(model) .^ 2
    lev = leverage(model)

    mse = mean(sqrerr)
    return @. sqrerr / (p * mse) * lev / (1 - lev)^2
end
