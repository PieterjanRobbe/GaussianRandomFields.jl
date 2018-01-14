## exponential.jl : implementation of exponential covariance function

## Exponential ##
struct Exponential{T} <: CovarianceStructure{T}
    λ::T
    σ::T
    p::T
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
function Exponential(λ::T where {T<:Real}; σ=1.0::T where {T<:Real}, p=2::T where {T<:Real}) 
    λ > 0 || throw(ArgumentError("correlation length λ of exponential covariance cannot be negative or zero"))
    σ > 0 || throw(ArgumentError("marginal standard deviation σ of exponential covariance cannot be negative or zero"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1"))
    Exponential{promote_type(typeof(λ),typeof(σ),typeof(p))}(promote(λ,σ,p)...) 
end

# evaluate exponential covariance
function apply(e::Exponential,x::T) where {T<:Real}
    exp(-x/e.λ)
end

show(io::IO, e::Exponential) = print(io, "exponential (λ=$(e.λ), σ=$(e.σ), p=$(e.p))")
