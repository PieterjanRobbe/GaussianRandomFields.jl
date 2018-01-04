# TODO jldoctest
## exponential.jl : implementation of exponential covariance function

struct Exponential{T} <: CovarianceStructure{T}
    λ::T
    σ::T
    p::T
end
    
"""
`Exponential(λ; σ=1., p=2)`

Create an exponential covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm.

Examples:
```
e = Exponential(0.1)
julia> exponential covariance structure with correlation length λ = 0.1, marginal standard deviation σ = 1.0 and 2.0-norm

e = Exponential(0.1, σ=2)
julia> exponential covariance structure with correlation length λ = 0.1, marginal standard deviation σ = 2.0 and 2.0-norm
```
"""
function Exponential(λ::T where {T<:Real};σ=1.0::T where {T<:Real},p=2::T where {T<:Real}) 
    λ > 0 || throw(ArgumentError("correlation length λ of exponential covariance cannot be negative or zero!"))
    σ > 0 || throw(ArgumentError("marginal standard deviation σ of exponential covariance cannot be negative or zero!"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1!"))
    Exponential{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ,σ,p)...) 
end

function apply(e::Exponential,x::T) where {T<:Real}
    exp(-x/e.λ)
end

show(io::IO,e::Exponential) = print(io, "exponential covariance structure with correlation length λ = $(e.λ), marginal standard deviation σ = $(e.σ) and $(e.p)-norm")
