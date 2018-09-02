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
    SquaredExponential(λ, σ=1, p=2)

Create a squared exponential (Gaussian) covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm.

# Examples
```jldoctest
julia> s1 = SquaredExponential(0.1)
Gaussian (λ=0.1, σ=1.0, p=2.0)

julia> s2 = SquaredExponential(0.1, σ=2.)
Gaussian (λ=0.1, σ=2.0, p=2.0)

```
"""
SquaredExponential(λ::Real; σ::Real=1.0, p::Real=2) =
    SquaredExponential{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ, σ, p)...)

"""
    Gaussian(λ, σ=1, p=2)

Create a Gaussian (squared exponential) covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm.

# Examples
```jldoctest
julia> g1 = Gaussian(0.1)
Gaussian (λ=0.1, σ=1.0, p=2.0)

julia> g2 = Gaussian(0.1, σ=2.)
Gaussian (λ=0.1, σ=2.0, p=2.0)

```
"""
const Gaussian = SquaredExponential

# evaluate Gaussian covariance
apply(s::SquaredExponential, x::Real) = exp(-(x / s.λ)^2)

show(io::IO, s::SquaredExponential) = print(io, "Gaussian (λ=$(s.λ), σ=$(s.σ), p=$(s.p))")
