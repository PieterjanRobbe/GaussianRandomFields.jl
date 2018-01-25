## matern.jl : implementation of Mat\'ern covariance function

## Matern ##
struct Matern{T} <: IsotropicCovarianceStructure{T}
    λ::T
    ν::T
    σ::T
    p::T
end

"""
    Matern(λ, ν, σ=1, p=2)

Create a Mat\u00E9rn covariance structure with correlation length `λ`, smoothness `ν`, (optional) marginal standard deviation `σ` and (optional) `p`-norm.

# Examples
```jldoctest
julia> m1 = Matern(0.1, 1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> m2 = Matern(0.1, 1.0, σ=2.0)
Matérn (λ=0.1, ν=1.0, σ=2.0, p=2.0)

```
"""
function Matern(λ::T where {T<:Real}, ν::T where {T<:Real}; σ=1.0::T where {T<:Real}, p=2::T where {T<:Real})
    λ > 0 || throw(ArgumentError("correlation length λ of Mat\u00E9rn covariance cannot be negative or zero"))
    ν > 0 || throw(ArgumentError("smoothness ν of Mat\u00E9rn covariance cannot be negative or zero"))
    σ > 0 || throw(ArgumentError("marginal standard deviation σ of Mat\u00E9rn covariance cannot be negative or zero"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1"))
	isinf(p) && throw(ArgumentError("in p-norm, p cannot be infinity"))
    Matern{promote_type(typeof(λ),typeof(ν),typeof(σ),typeof(p))}(promote(λ,ν,σ,p)...) 
end

# TODO : Evaluation of the besselk function is extremely expensive; maybe try to catch special cases for ν = 1/2, 3/2 etc.
# evaluate Matern covariance
function apply(m::Matern,x::T) where {T<:Real}
    x == zero(T) ? one(T) : 2^(1-m.ν)/gamma(m.ν)*(sqrt(2*m.ν)*x/m.λ)^m.ν*besselk(m.ν,sqrt(2*m.ν)*x/m.λ)
end

show(io::IO, m::Matern) = print(io, "Mat\u00E9rn (λ=$(m.λ), ν=$(m.ν), σ=$(m.σ), p=$(m.p))")
