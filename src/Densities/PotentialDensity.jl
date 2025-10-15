# ? PotentialDensity (supply a potential, get a NON_NORMALIZED density)
export PotentialDensity

struct PotentialDensity{D, N, doAd} <: AbstractDensity{D, N, doAd}
    potential::D
end
PotentialDensity{N}(potential::D) where {D, N} = PotentialDensity{D, N, true}(potential)

capabilities(::Type{<:PotentialDensity}) = LogDensityProblems.LogDensityOrder{1}()

potential(D::PotentialDensity) = D.potential
logdensity(D::PotentialDensity) = (-) ∘ potential(D)
gradpotential(D::PotentialDensity) = (.-) ∘ gradlogdensity(D)
density(D::PotentialDensity) = exp ∘ (-) ∘ potential(D)
