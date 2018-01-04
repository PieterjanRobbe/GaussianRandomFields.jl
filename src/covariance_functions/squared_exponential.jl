# TODO jldoctest
## squared_exponential.jl : implementation of squared exponential covariance function

struct SquaredExponential{T} <: CovarianceStructure{T}
    λ::T
    σ::T
    p::T
end
    
"""
`SquaredExponential(λ; σ=1., p=2)`

Create a squared exponential covariance structure with correlation length `λ`, (optional) marginal standard deviation `σ` and (optional) `p`-norm.

Examples:
```
s = SquaredExponential(0.1)
julia> squared exponential covariance structure with correlation length λ = 0.1, marginal standard deviation σ = 1.0 and 2.0-norm 

s = SquaredExponential(0.1, σ=2.)
julia> squared exponential covariance structure with correlation length λ = 0.1, marginal standard deviation σ = 2.0 and 2.0-norm 
```
"""
function SquaredExponential(λ::T where {T<:Real};σ=1.0::T where {T<:Real},p=2::T where {T<:Real}) 
    λ > 0 || throw(ArgumentError("correlation length λ of squared exponential covariance cannot be negative or zero!"))
    σ > 0 || throw(ArgumentError("marginal standard deviation σ of squared exponential covariance cannot be negative or zero!"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1!"))
    SquaredExponential{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ,σ,p)...) 
end

function apply(s::SquaredExponential,x::T) where {T<:Real}
    exp(-(x/s.λ)^2)
end

show(io::IO,s::SquaredExponential) = print(io, "squared exponential covariance structure with correlation length λ = $(s.λ), marginal standard deviation σ = $(s.σ) and $(s.p)-norm")
