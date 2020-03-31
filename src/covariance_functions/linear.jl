## linear.jl : implementation of linear covariance function

## Linear ##
struct Linear{T} <: IsotropicCovarianceStructure{T}
    λ::T
    σ::T
    p::T

    function Linear{T}(λ::T, σ::T, p::T) where T
        λ > 0 || throw(DomainError(λ, "correlation length λ of linear covariance cannot be negative or zero"))
        σ > 0 || throw(DomainError(σ, "marginal standard deviation σ of linear covariance cannot be negative or zero"))
        p >= 1 || throw(DomainError(p, "in p-norm, p must be greater than or equal to 1"))
        isinf(p) && throw(DomainError(p, "in p-norm, p cannot be infinity"))

        new{T}(λ, σ, p)
    end
end

"""
    Linear(λ, [σ = 1], [p = 2])

Linear covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm, defined as

``C(x, y) = \\begin{cases} σ \\left(1 - \\displaystyle\\frac{ρ}{λ}\\right) & \\text{if }ρ ≤ λ\\\\ 0 & \\text{if }ρ>λ\\end{cases}``

with ``ρ = ||x - y||_p``.

# Examples
```jldoctest
julia> Linear(0.1)
linear (λ=0.1, σ=1.0, p=2.0)

julia> Linear(1.0, σ=2)
linear (λ=1.0, σ=2.0, p=2.0)

```
See also: [`Exponential`](@ref), [`Spherical`](@ref), [`Whittle`](@ref), [`Gaussian`](@ref), [`SquaredExponential`](@ref), [`Matern`](@ref)
"""
Linear(λ::Real; σ::Real=1.0, p::Real=2) = Linear{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ, σ, p)...)

# evaluate exponential covariance
apply(l::Linear, x::Real) = max(zero(x), 1 - x / l.λ)

# short name
shortname(::Linear) = "linear"
