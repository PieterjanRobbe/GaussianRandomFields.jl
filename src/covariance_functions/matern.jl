## matern.jl : implementation of Mat\'ern covariance function?

## Matern ##
struct Matern{T} <: IsotropicCovarianceStructure{T}
    λ::T
    ν::T
    σ::T
    p::T

    function Matern{T}(λ::T, ν::T, σ::T, p::T) where T
        λ > 0 || throw(DomainError(λ, "correlation length λ of Mat\u00E9rn covariance cannot be negative or zero"))
        ν > 0 || throw(DomainError(ν, "smoothness ν of Mat\u00E9rn covariance cannot be negative or zero"))
        σ > 0 || throw(DomainError(σ, "marginal standard deviation σ of Mat\u00E9rn covariance cannot be negative or zero"))
        p >= 1 || throw(DomainError(p, "in p-norm, p must be greater than or equal to 1"))
    	isinf(p) && throw(DomainError(p, "in p-norm, p cannot be infinity"))

        new{T}(λ, ν, σ, p)
    end
end

"""
    Matern(λ, ν, [σ = 1], [p = 2])

Mat\u00E9rn covariance structure with correlation length `λ`, smoothness `ν`, (optional) marginal standard deviation `σ` and (optional) `p`-norm, defined as

``C(x, y) = σ \\displaystyle\\frac{2^{1 - ν}}{Γ(ν)} \\left(\\frac{\sqrt(2\nu)ρ}{λ}\\right)^ν K_ν\\left(\\frac{\sqrt(2\nu)ρ}{λ}\\right)``

with ``ρ = ||x - y||_p``.

# Examples
```jldoctest
julia> Matern(0.1, 1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> Matern(1, 1, σ=2.0)
Matérn (λ=1.0, ν=1.0, σ=2.0, p=2.0)

```
See also: [`Exponential`](@ref), [`Linear`](@ref), [`Spherical`](@ref), [`Whittle`](@ref), [`Gaussian`](@ref), [`SquaredExponential`](@ref)
"""
Matern(λ::Real, ν::Real; σ::Real=1.0, p::Real=2) =
    Matern{promote_type(typeof(λ),typeof(ν),typeof(σ),typeof(p))}(promote(λ, ν, σ, p)...)

# TODO : Evaluation of the besselk function is extremely expensive; maybe try to catch special cases for ν = 1/2, 3/2 etc.
# evaluate Matern covariance
function apply(m::Matern, x::Real)
    if iszero(x)
        float(one(x))
    else
        2^(1 - m.ν) / gamma(m.ν) * (sqrt(2 * m.ν) * x / m.λ)^m.ν * besselk(m.ν, sqrt(2 * m.ν) * x / m.λ)
    end
end

# short name
shortname(::Matern) = "Mat\u00E9rn"
