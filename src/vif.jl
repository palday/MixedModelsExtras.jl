# I think car computes GVIF combining all the column-wise VIFs
# for a single categorical predictor, so that's something
# to look into

"""
    vifnames(m::MixedModel)

Show the names of the coefficients for which a VIF is computed.

See also [`vif`](@ref).
"""
function vifnames(m::MixedModel)
    cn = coefnames(m)
    first(cn) == "(Intercept)" ||
        throw(ArgumentError("VIF only defined for models with an intercept term"))
    cn = @view cn[2:end]
    length(cn) > 1 ||
        throw(ArgumentError("VIF not meaningful for models with only one non-intercept term"))

    return cn
end

"""
    vif(m::MixedModel)

Compute the variance inflation factor (VIF).

See also [`vifnames`](@ref).

!!! warning
    This method will fail if there is (numerically) perfect multicollinearity,
    i.e. rank deficiency (in the fixed effects). In that case though, the VIF
    isn't particularly informative anyway.
"""
function vif(m::MixedModel)
    mm = vcov(m; corr=true)
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
