# I think car computes GVIF combining all the column-wise VIFs
# for a single categorical predictor, so that's something
# to look into

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
termnames(model) = collect(_rename_intercept.(string.(formula(model).rhs.terms)))
function termnames(model::MixedModel)
    return collect(_rename_intercept.(string.(formula(model).rhs[1].terms)))
end

_terms(model) = collect(formula(model).rhs.terms)
_terms(model::MixedModel) = collect(formula(model).rhs[1].terms)

"""
    vif(m::RegressionModel)

Compute the variance inflation factor (VIF).

Returns a vector of inflation factors computed for each coefficient except
for the intercept.
In other words, the corresponding coefficients are `coefnames(m)[2:end]`.

See also [`coefnames`](@ref).

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
    # NB: The coorrelation matrix is positive definite and hence invertible
    #     unless there is perfect rank deficiency, hence the warning.
    return diag(inv(Symmetric(mm)))
end

"""
    gvif(m::RegressionModel; scaled_by_df=false)

Compute the generalized variance inflation factor (GVIF).

If `scale_by_df=true`, then each GVIF is scaled by the degrees of freedom
for (number of coefficients associated with) the predictor: ``GVIF^(1 / (2*df))``

Returns a vector of inflation factors computed for each term except
for the intercept.
In other words, the corresponding coefficients are `termnames(m)[2:end]`.

See also [`termnames`](@ref).

!!! warning
    This method will fail if there is (numerically) perfect multicollinearity,
    i.e. rank deficiency (in the fixed effects). In that case though, the VIF
    isn't particularly informative anyway.
"""
function gvif(m::RegressionModel; scaled_by_df=false)
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

    return scaled_by_df ? vals .^ (1 ./ (2 .* df)) : vals
end
