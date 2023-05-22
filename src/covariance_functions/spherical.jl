## spherical.jl : implementation of spherical covariance function

## Spherical ##
struct Spherical{T} <: IsotropicCovarianceStructure{T}
    λ::T
    σ::T
    p::T

    function Spherical{T}(λ::T, σ::T, p::T) where T
        λ > 0 || throw(DomainError(λ, "correlation length λ of spherical covariance cannot be negative or zero"))
        σ > 0 || throw(DomainError(σ, "marginal standard deviation σ of spherical covariance cannot be negative or zero"))
        p >= 1 || throw(DomainError(p, "in p-norm, p must be greater than or equal to 1"))
        isinf(p) && throw(DomainError(p, "in p-norm, p cannot be infinity"))

        new{T}(λ, σ, p)
    end
end

"""
    Spherical(λ, [σ = 1], [p = 2])

Spherical covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm, defined as

``C(x, y) = \\begin{cases} σ \\left(1 - \\displaystyle\\frac{3}{2}\\frac{ρ}{λ} + \\frac{1}{2}\\left(\\frac{ρ}{λ}\\right)^3\\right) & \\text{for }ρ≤λ\\\\0 & \\text{for }ρ>λ\\end{cases}``

with ``ρ = ||x - y||_p``.

# Examples
```jldoctest
julia> Spherical(0.1)
spherical (λ=0.1, σ=1.0, p=2.0)

julia> Spherical(1.0, σ=2)
spherical (λ=1.0, σ=2.0, p=2.0)

```
See also: [`Exponential`](@ref), [`Linear`](@ref), [`Whittle`](@ref), [`Gaussian`](@ref), [`SquaredExponential`](@ref), [`Matern`](@ref)
"""
Spherical(λ::Real; σ::Real=1.0, p::Real=2) = Spherical{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ, σ, p)...)

# evaluate spherical covariance
apply(s::Spherical, x::Real) =s.σ * s.σ * max(zero(x), 1 - 3/2 * x/s.λ + 1/2 * (x/s.λ)^3)

# short name
shortname(::Spherical) = "spherical"
