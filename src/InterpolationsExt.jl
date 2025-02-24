import Interpolations

const AbstractInterpolationDensity{D, doAd} = AbstractDensity{D,
                                                              doAd} where {
                                                                           D <:
                                                                           Interpolations.AbstractInterpolation,
                                                                           doAd}
function (D::AbstractInterpolationDensity)(x::AbstractVector)
    d = D.distribution(x...)
    if d < 0
        @warn "Density ($(round(d; sigdigits=2))) is negative at $x. Clipping to `eps`"
    end
    clamp(d, eps(), Inf)
end
function Density(d::D) where {D <: Interpolations.AbstractInterpolation}
    Density{true}(d)
end
function LogDensityProblems.capabilities(::Type{<:AbstractInterpolationDensity})
    LogDensityProblems.LogDensityOrder{1}()
end
function LogDensityProblems.dimension(D::AbstractInterpolationDensity)
    ndims(distribution(D))
end
function LogDensityProblems.logdensity(D::AbstractInterpolationDensity, x)
    D(x) |> log
end
function logdensity_and_gradient(D::AbstractDistributionDensity, x)
    (LogDensityProblems.logdensity(D, x), gradlogdensity(D, x))
end

function gradlogdensity(D::AbstractInterpolationDensity, x)
    _gradlogdensity(D, x)
end

function vignette(sz::Tuple{Int, Int}, radius = (0.75, 0.75))
    nrows, ncols = sz
    fr, fc = radius
    cr, cc = (nrows + 1) / 2, (ncols + 1) / 2
    rn = abs.(collect(1:nrows) .- cr) ./ (nrows / 2)
    cn = abs.(collect(1:ncols) .- cc) ./ (ncols / 2)
    rw = ifelse.(rn .< (1 - fr), 1.0,
                 ifelse.(rn .> 1.0, 0.0, 1 .- (rn .- (1 - fr)) ./ fr))
    cw = ifelse.(cn .< (1 - fc), 1.0,
                 ifelse.(cn .> 1.0, 0.0, 1 .- (cn .- (1 - fc)) ./ fc))
    return rw * cw'
end
vignette(sz::Tuple{Int, Int}, radius::Number) = vignette(sz, (radius, radius))
vignette(sz::AbstractMatrix, args...) = vignette(size(sz), args...)
