## exponential.jl : implementation of exponential covariance function

## Exponential ##
struct Exponential{T} <: IsotropicCovarianceStructure{T}
    λ::T
    σ::T
    p::T

    function Exponential{T}(λ::T, σ::T, p::T) where T
        λ > 0 || throw(DomainError(λ, "correlation length λ of exponential covariance cannot be negative or zero"))
        σ > 0 || throw(DomainError(σ, "marginal standard deviation σ of exponential covariance cannot be negative or zero"))
        p >= 1 || throw(DomainError(p, "in p-norm, p must be greater than or equal to 1"))
        isinf(p) && throw(DomainError(p, "in p-norm, p cannot be infinity"))

        new{T}(λ, σ, p)
    end
end

"""
    Exponential(λ, σ=1, p=2)

Create an exponential covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm.

# Examples
```jldoctest
julia> e1 = Exponential(0.1)
exponential (λ=0.1, σ=1.0, p=2.0)

julia> e2 = Exponential(0.1, σ=2)
exponential (λ=0.1, σ=2.0, p=2.0)

```
"""
Exponential(λ::Real; σ::Real=1.0, p::Real=2) = Exponential{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ, σ, p)...)

# evaluate exponential covariance
apply(e::Exponential, x::Real) = exp(-x / e.λ)

show(io::IO, e::Exponential) = print(io, "exponential (λ=$(e.λ), σ=$(e.σ), p=$(e.p))")
