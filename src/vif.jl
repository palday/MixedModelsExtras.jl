_rename_intercept(s) = strip(s) == "1" ? "(Intercept)" : s

"""
    termnames(model)

Return the names associated with (fixed effects) terms in a model.

For models with only continuous predictors, this is the same as
`coefnames(model)`.
For models with categorical predictors, the returned names reflect
the categorical predictor and not the coefficients resulting from
the choice of contrast coding.
"""
termnames(model) = _rename_intercept.(string.(_terms(model)))

_terms(model) = collect(formula(model).rhs.terms)
_terms(model::MixedModel) = collect(formula(model).rhs[1].terms)

"""
    vif(m::RegressionModel)

Compute the variance inflation factor (VIF).

Returns a vector of inflation factors computed for each coefficient except
for the intercept.
In other words, the corresponding coefficients are `coefnames(m)[2:end]`.

The [variance inflation factor (VIF)](https://en.wikipedia.org/wiki/Variance_inflation_factor) measures
the increase in the variance of a parameter's estimate in a model with multiple parameters relative to
the variance in of a paremeter's estimate in a model containing only that parameter.

See also [`coefnames`](@ref), [`gvif`](@ref).

!!! warning
    This method will fail if there is (numerically) perfect multicollinearity,
    i.e. rank deficiency (in the fixed effects). In that case though, the VIF
    isn't particularly informative anyway.
"""
function vif(m::RegressionModel)
    mm = StatsBase.cov2cor!(vcov(m), stderror(m))
    all(==(1), view(modelmatrix(m), :, 1)) ||
        throw(ArgumentError("VIF only defined for models with an intercept term"))
    mm = @view mm[2:end, 2:end]
    size(mm, 2) > 1 ||
        throw(ArgumentError("VIF not meaningful for models with only one non-intercept term"))
    # directly computing inverses is bad, but
    # generally this shouldn't be a huge matrix and it's symmetric
    # NB: The correlation matrix is positive definite and hence invertible
    #     unless there is perfect rank deficiency, hence the warning.
    # NB: The determinate technique for GVIF could also be applied
    #     columnwise (instead of Term-wise) here, but it wouldn't really
    #     be any more efficient because this is a Symmetric matrix and computing
    #     all those determinants has its cost. The determinants also hint at
    #     how you could show equivalency, if you remember that inversion is solving
    #     a linear system and that Cramer's rule -- which uses determinants --
    #     can also a linear system
    return diag(inv(Symmetric(mm)))
end

"""
    gvif(m::RegressionModel; scale=false)

Compute the generalized variance inflation factor (GVIF).

If `scale=true`, then each GVIF is scaled by the degrees of freedom
for (number of coefficients associated with) the predictor: ``GVIF^(1 / (2*df))``

Returns a vector of inflation factors computed for each term except
for the intercept.
In other words, the corresponding coefficients are `termnames(m)[2:end]`.

The [generalized variance inflation factor (VIF)](https://doi.org/10.2307/2290467)
measures the increase in the variance of a (group of) parameter's estimate in a model
with multiple parameters relative to the variance of a parameter's estimate in a
model containing only that parameter. For continuous, numerical predictors, the GVIF
is the same as the VIF, but for categorical predictors, the GVIF provides a single
number for the entire group of contrast-coded coefficients associated with a categorical
predictor.

See also [`termnames`](@ref), [`vif`](@ref).

!!! warning
    This method will fail if there is (numerically) perfect multicollinearity,
    i.e. rank deficiency (in the fixed effects). In that case though, the VIF
    isn't particularly informative anyway.

## References

Fox, J., & Monette, G. (1992). Generalized Collinearity Diagnostics.
Journal of the American Statistical Association, 87(417), 178. doi:10.2307/2290467
"""
function gvif(m::RegressionModel; scale=false)
    mm = StatsBase.cov2cor!(vcov(m), stderror(m))

    all(==(1), view(modelmatrix(m), :, 1)) ||
        throw(ArgumentError("GVIF only defined for models with an intercept term"))
    mm = @view mm[2:end, 2:end]
    size(mm, 2) > 1 ||
        throw(ArgumentError("GVIF not meaningful for models with only one non-intercept term"))

    tn = @view termnames(m)[2:end]
    trms = @view _terms(m)[2:end]

    df = width.(trms)
    vals = zeros(eltype(mm), length(tn))
    detmm = det(mm)
    acc = 0
    for idx in axes(vals, 1)
        wt = df[idx]
        trm_only = [acc < i <= (acc + wt) for i in axes(mm, 2)]
        trm_excl = .!trm_only
        vals[idx] = det(Symmetric(view(mm, trm_only, trm_only))) *
                    det(Symmetric(view(mm, trm_excl, trm_excl))) /
                    detmm
        acc += wt
    end

    if scale
        vals .= vals .^ (1 ./ (2 .* df))
    end
    return vals
end
