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
    # NB: The correlation matrix is positive definite and hence invertible
    #     unless there is perfect rank deficiency, hence the warning.
    # NB: The determinate technique for GVIF could also be applied
    #     columnwise (instead of Term-wise) here, but it wouldn't really
    #     be any more efficient because this is a Symmetric matrix and computing
    #     all those determinants has its cost. The determinants also hint at
    #     how you could show equivalency, if you remember that inversion is solving
    #     a linear system and that Cramer's rule -- which uses determinants --
    #     can also a linear system
    # so we want diag(inv(mm)) but directly computing inverses is bad.
    # well we can also take advantage of the fact that inv(mm) == (mm') ./ det(mm)
    # and since this matrix is symmetric and we only care about the diagonal
    # we can rewrite that as:
    return diag(mm) ./ det(mm)
    # benchmarks for different ways:
    # julia> @benchmark vif($(lm1)) # diag(inv(mm))
    # BenchmarkTools.Trial: 10000 samples with 10 evaluations.
    # Range (min … max):  1.120 μs … 924.783 μs  ┊ GC (min … max): 0.00% … 99.61%
    # Time  (median):     1.252 μs               ┊ GC (median):    0.00%
    # Time  (mean ± σ):   1.382 μs ±   9.238 μs  ┊ GC (mean ± σ):  6.67% ±  1.00%

    #         █▇▃
    # ▃█▇▄▂▂▂████▆▄▄▆▆▆▄▄▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
    # 1.12 μs         Histogram: frequency by time        1.91 μs <

    # Memory estimate: 1.20 KiB, allocs estimate: 13.

    # julia> @benchmark vif($(lm1)) # diag(inv(Symmetric(mm)))
    # BenchmarkTools.Trial: 10000 samples with 10 evaluations.
    # Range (min … max):  1.173 μs …  1.014 ms  ┊ GC (min … max): 0.00% … 99.56%
    # Time  (median):     1.310 μs              ┊ GC (median):    0.00%
    # Time  (mean ± σ):   1.463 μs ± 10.132 μs  ┊ GC (mean ± σ):  6.90% ±  1.00%

    # ▄  ▇█
    # ▆█▄▂███▆▇▆▄▃▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
    # 1.17 μs        Histogram: frequency by time        2.63 μs <

    # Memory estimate: 1.30 KiB, allocs estimate: 15.

    # julia> @benchmark vif($(lm1)) # diag(mm) ./ det(mm)
    # BenchmarkTools.Trial: 10000 samples with 10 evaluations.
    # Range (min … max):  1.107 μs … 934.525 μs  ┊ GC (min … max): 0.00% … 99.63%
    # Time  (median):     1.245 μs               ┊ GC (median):    0.00%
    # Time  (mean ± σ):   1.364 μs ±   9.337 μs  ┊ GC (mean ± σ):  6.82% ±  1.00%

    #             ▄█▆▃
    # ▂▅▆▅▃▂▂▂▂▅█████▆▄▄▅▆▆▆▅▄▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
    # 1.11 μs         Histogram: frequency by time        1.72 μs <

    # Memory estimate: 1.14 KiB, allocs estimate: 12.

    # julia> @benchmark vif($(lm1))  diag(mm) ./ det(Symmetric(mm))
    # BenchmarkTools.Trial: 10000 samples with 10 evaluations.
    # Range (min … max):  1.162 μs …  1.010 ms  ┊ GC (min … max): 0.00% … 99.58%
    # Time  (median):     1.301 μs              ┊ GC (median):    0.00%
    # Time  (mean ± σ):   1.434 μs ± 10.089 μs  ┊ GC (mean ± σ):  7.01% ±  1.00%

    # ▁▂     ▅█▅▁
    # ▂██▆▃▂▂▄████▆▄▅▇▆▅▄▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
    # 1.16 μs        Histogram: frequency by time        1.95 μs <

    # Memory estimate: 1.28 KiB, allocs estimate: 14.

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
    # benchmarking shows thad adding in Symmetric() here before det()
    # actually slows things down a bit
    detmm = det(mm)
    acc = 0
    for idx in axes(vals, 1)
        wt = df[idx]
        trm_only = [acc < i <= (acc + wt) for i in axes(mm, 2)]
        trm_excl = .!trm_only
        vals[idx] = det(view(mm, trm_only, trm_only)) *
                    det(view(mm, trm_excl, trm_excl)) /
                    detmm
        acc += wt
    end

    if scale
        vals .= vals .^ (1 ./ (2 .* df))
    end
    return vals
end
