function vifnames(m::MixedModel)
    cn = coefnames(m)
    first(cn) == "(Intercept)" ||
        throw(ArgumentError("VIF only defined for models with an intercept term"))
    cn = @view cn[2:end]
    length(cn) > 1 ||
        throw(ArgumentError("VIF not meaningful for models with only one non-intercept term"))

    return cn
end


function vif(m::MixedModel)
    mm = vcov(m; corr=true)
    all(==(1), view(modelmatrix(m), :, 1)) ||
        throw(ArgumentError("VIF only defined for models with an intercept term"))
    mm = @view mm[:, 2:end]
    size(mm, 2) > 1 ||
        throw(ArgumentError("VIF not meaningful for models with only one non-intercept term"))
    # directly computing inverses is bad, but
    # generally this shouldn't be a huge matrix and it's symmetric
    return diag(inv(Symmetric(mm)))
end