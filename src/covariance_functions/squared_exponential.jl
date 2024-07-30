## squared_exponential.jl : implementation of squared exponential covariance function

## SquaredExponential ##
struct SquaredExponential{T} <: IsotropicCovarianceStructure{T}
    λ::T
    σ::T
    p::T

    function SquaredExponential{T}(λ::T, σ::T, p::T) where T
        λ > 0 || throw(DomainError(λ, "correlation length λ of squared exponential covariance cannot be negative or zero"))
        σ > 0 || throw(DomainError(σ, "marginal standard deviation σ of squared exponential covariance cannot be negative or zero"))
        p >= 1 || throw(DomainError(p, "in p-norm, p must be greater than or equal to 1"))
        isinf(p) && throw(DomainError(p, "in p-norm, p cannot be infinity"))

        new{T}(λ, σ, p)
    end
end

"""
    SquaredExponential(λ, [σ = 1], [p = 2])

Squared exponential (Gaussian) covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm, defined as

``C(x, y) = σ^2 \\exp\\left(-\\left(\\displaystyle\\frac{ρ}{λ}\\right)^2\\right)``

with ``ρ = ||x - y||_p``.

# Examples
```jldoctest
julia> SquaredExponential(0.1)
Gaussian (λ=0.1, σ=1.0, p=2.0)

julia> SquaredExponential(1, σ=2.)
Gaussian (λ=1.0, σ=2.0, p=2.0)

```
See also: [`Exponential`](@ref), [`Linear`](@ref), [`Spherical`](@ref), [`Whittle`](@ref), [`Gaussian`](@ref), [`Matern`](@ref)
"""
SquaredExponential(λ::Real; σ::Real=1.0, p::Real=2) =
    SquaredExponential{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ, σ, p)...)

"""
    Gaussian(λ, [σ = 1], [p = 2])

Gaussian (squared exponential) covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm, defined as

``C(x, y) = σ \\exp\\left(-\\left(\\displaystyle\\frac{ρ}{λ}\\right)^2\\right)``

with ``ρ = ||x - y||_p``.

# Examples
```jldoctest
julia> Gaussian(0.1)
Gaussian (λ=0.1, σ=1.0, p=2.0)

julia> Gaussian(1, σ=2.)
Gaussian (λ=1.0, σ=2.0, p=2.0)

```
See also: [`Exponential`](@ref), [`Linear`](@ref), [`Spherical`](@ref), [`Whittle`](@ref), [`SquaredExponential`](@ref), [`Matern`](@ref)
"""
const Gaussian = SquaredExponential

# evaluate Gaussian covariance
apply(s::SquaredExponential, x::Real) = s.σ * s.σ * exp(-(x / s.λ)^2)

# short name
shortname(::SquaredExponential) = "Gaussian"
