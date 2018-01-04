# TODO jldoctest
## matern.jl : implementation of Mat\'ern covariance function

struct Matern{T} <: CovarianceStructure{T}
    λ::T
    σ::T
    ν::T
    p::T
end

"""
`Matern(λ, ν; σ=1., p=2)`

Create a Mat\u00E9rn covariance structure with correlation length `λ`, smoothness `ν`, (optional) marginal standard deviation `σ` and (optional) `p`-norm.

Examples:
```
m = Matern(0.1, 1.0)
julia> Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0, marginal standard deviation σ = 1.0 and 1.0-norm

m = Matern(0.1, 1.0, σ=2.0)
julia> Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0, marginal standard deviation σ = 2.0 and 1.0-norm
```
"""
function Matern(λ::T where {T<:Real},ν::T where {T<:Real};σ=1.0::T where {T<:Real},p=2::T where {T<:Real})
    λ > 0 || throw(ArgumentError("correlation length λ of Mat\u00E9rn covariance cannot be negative or zero!"))
    ν > 0 || throw(ArgumentError("smoothness ν of Mat\u00E9rn covariance cannot be negative or zero!"))
    σ > 0 || throw(ArgumentError("marginal standard deviation σ of Mat\u00E9rn covariance cannot be negative or zero!"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1!"))
    Matern{promote_type(typeof(λ),typeof(ν),typeof(σ),typeof(p))}(promote(λ,ν,σ,p)...) 
end

# NOTE TODO
# Evaluation of the besselk function is extremely expensive; maybe try to catch special cases for ν = 1/2, 3/2 etc.
function apply(m::Matern,x::T) where {T<:Real}
    x == zero(T) ? one(T) : 2^(1-m.ν)/gamma(m.ν)*(sqrt(2*m.ν)*x/m.λ)^m.ν*besselk(m.ν,sqrt(2*m.ν)*x/m.λ)
end

show(io::IO,m::Matern) = print(io, "Mat\u00E9rn covariance structure with correlation length λ = $(m.λ), smoothness ν = $(m.ν), marginal standard deviation σ = $(m.σ) and $(m.p)-norm")
