## Whittle.jl : implementation of Whittle covariance function

## Whittle ##
struct Whittle{T} <: IsotropicCovarianceStructure{T}
    λ::T
    σ::T
    p::T

    function Whittle{T}(λ::T, σ::T, p::T) where T
        λ > 0 || throw(DomainError(λ, "correlation length λ of Whittle covariance cannot be negative or zero"))
        σ > 0 || throw(DomainError(σ, "marginal standard deviation σ of Whittle covariance cannot be negative or zero"))
        p >= 1 || throw(DomainError(p, "in p-norm, p must be greater than or equal to 1"))
        isinf(p) && throw(DomainError(p, "in p-norm, p cannot be infinity"))

        new{T}(λ, σ, p)
    end
end

"""
    Whittle(λ, [σ = 1], [p = 2])

Whittle covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm, defined as
    
``C(x, y) = σ \\displaystyle\\frac{ρ}{λ} K₁\\left(\\frac{ρ}{λ}\\right)``

with ``ρ = ||x-y||_p``.

# Examples
```jldoctest
julia> Whittle(0.1)
Whittle (λ=0.1, σ=1.0, p=2.0)

julia> Whittle(1.0, σ=2)
Whittle (λ=1.0, σ=2.0, p=2.0)

```
See also: [`Exponential`](@ref), [`Linear`](@ref), [`Spherical`](@ref), [`Gaussian`](@ref), [`SquaredExponential`](@ref), [`Matern`](@ref)
"""
Whittle(λ::Real; σ::Real=1.0, p::Real=2) = Whittle{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ, σ, p)...)

# evaluate Whittle covariance
function apply(w::Whittle, x::Real)
    if iszero(x)
        w.σ * w.σ * float(one(x))
    else
        ρ = x / w.λ
        w.σ * w.σ * ρ * besselk(1, ρ)
    end
end

# short name
shortname(::Whittle) = "Whittle"
