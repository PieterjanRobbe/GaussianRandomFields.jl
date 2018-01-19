## squared_exponential.jl : implementation of squared exponential covariance function

## SquaredExponential ##
struct SquaredExponential{T} <: CovarianceStructure{T}
    λ::T
    σ::T
    p::T
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
function SquaredExponential(λ::T where {T<:Real}; σ=1.0::T where {T<:Real}, p=2::T where {T<:Real}) 
    λ > 0 || throw(ArgumentError("correlation length λ of squared exponential covariance cannot be negative or zero"))
    σ > 0 || throw(ArgumentError("marginal standard deviation σ of squared exponential covariance cannot be negative or zero"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1"))
	isinf(p) && throw(ArgumentError("in p-norm, p cannot be infinity"))
    SquaredExponential{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ,σ,p)...) 
end

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
function apply(s::SquaredExponential,x::T) where {T<:Real}
    exp(-(x/s.λ)^2)
end

show(io::IO, s::SquaredExponential) = print(io, "Gaussian (λ=$(s.λ), σ=$(s.σ), p=$(s.p))")
