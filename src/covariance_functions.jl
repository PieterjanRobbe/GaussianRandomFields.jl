# TODO: evaluation of covariance function is crucial and bottleneck!!!
# TODO: complete evaluation of exponential/squared exponential

## covariance_functions.jl : GRF covariance functions

## CovarianceStructure ##
abstract type CovarianceStructure end

## CovarianceFunction ##
"""
`struct CovarianceFunction{d,T}`

Implements a covariance function of type `T` for a `d`-dimensional Gaussian random field.
"""
struct CovarianceFunction{d,T} 
    cov::T
end

apply(cov::CovarianceFunction,x::AbstractArray{T,N},y::AbstractArray{T,N}) where {T,N} = apply(cov.cov,x,y)

"""
`CovarianceFunction(d, cov)`

Create a covariance function in `d` dimensions for the covariance structure `cov`.

Examples:
```
julia> m = Matern(0.1,1.0,2)
Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0 and 2.0-norm

julia> c = CovarianceFunction(2,m)
2-dimensional covariance function with Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0 and 2.0-norm

```
"""
function CovarianceFunction(d::N where {N<:Integer},cov::T) where {T<:CovarianceStructure}
    d > 0 || throw(ArgumentError("dimension must be positive, got $(d)"))
    CovarianceFunction{d,T}(cov)
end

show(io::IO, c::CovarianceFunction{d}) where {d} = print(io, "$(d)-dimensional covariance function with $(c.cov)")

"""
`struct SeparableCovarianceFunction{d,V}`

Implements a separable covariance function of types `V` for a `d`-dimensional Gaussian random field.
Usefull for defining anisotropic random fields.
"""
struct SeparableCovarianceFunction{d,V}
    cov::V
end

"""
`SeparableCovarianceFunction(d, cov)`

Create a separable covariance function in `d` dimensions for the covariance structures `cov`. Usefull for defining anisotropic covariance functions.

Examples:
```
julia> m1 = Matern(0.1,1.0,2)
Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0 and 2.0-norm

julia> m2 = Matern(0.01,1.0,2)
Matérn covariance structure with correlation length λ = 0.01, smoothness ν = 1.0 and 2.0-norm

julia> c = SeparableCovarianceFunction(m1,m2)
2-dimensional separable covariance function with
  - Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0 and 2.0-norm  
  - Matérn covariance structure with correlation length λ = 0.01, smoothness ν = 1.0 and 2.0-norm

```
"""
SeparableCovarianceFunction(cov::Vector{T}) where {T<:CovarianceStructure} = SeparableCovarianceFunction{length(cov),Vector{T}}(cov)
SeparableCovarianceFunction(cov::Any...) = SeparableCovarianceFunction([cov...])

function show(io::IO, s::SeparableCovarianceFunction{d}) where {d}
    str = "$(d)-dimensional separable covariance function with \n"
    for elem in s.cov
        str *= "  - $(elem)\n"
    end
    print(io, str) 
end

## Matérn ##
struct Matern{T} <: CovarianceStructure
    λ::T
    ν::T
    p::T
end

"""
`Matern(λ, ν, p)`

Create a Mat\u00E9rn covariance structure with correlation length `λ`, smoothness `ν` and `p`-norm.

Examples:
```
m = Matern(0.1, 1.0, 1)
julia> Matérn covariance structure with correlation length λ = 0.1, smoothness ν = 1.0 and 1.0-norm

```
"""
function Matern(λ::T where {T<:Real},ν::T where {T<:Real},p::T where {T<:Real})
    λ > 0 || throw(ArgumentError("correlation length λ of Mat\u00E9rn covariance cannot be negative or zero!"))
    ν > 0 || throw(ArgumentError("smoothness ν of Mat\u00E9rn covariance cannot be negative or zero!"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1!"))
    Matern{promote_type(typeof(λ),typeof(ν),typeof(p))}(promote(λ,ν,p)...) 
end

# TODO: what if we make this into a matrix function? with dot bradcast?
function apply(m::Matern, x::AbstractArray{T,N}, y::AbstractArray{T,N}) where {T,N}
    C = zeros(T,size(x,1),size(y,1))
    for i in 1:size(x,1)
        for j in 1:size(y,1)
            @inbounds C[i,j] = all(x[i,:].==y[j,:]) ? 1 : 2^(1-m.ν)/gamma(m.ν)*(sqrt(2*m.ν)*norm(x[i,:]-y[j,:],m.p)/m.λ).^m.ν.*besselk(m.ν,sqrt(2*m.ν)*norm(x[i,:]-y[j,:],m.p)/m.λ)
        end
    end

    return C
end

show(io::IO,m::Matern) = print(io, "Mat\u00E9rn covariance structure with correlation length λ = $(m.λ), smoothness ν = $(m.ν) and $(m.p)-norm")


## Exponential ##
struct Exponential{T} <: CovarianceStructure
    λ::T
    p::T
end
    
"""
`Exponential(λ, p)`

Create an exponential covariance structure with correlation length `λ` and `p`-norm.

Examples:
```
e = Exponential(0.1, 2)
julia> exponential covariance structure with correlation length λ = 0.1 and 2.0-norm

```
"""
function Exponential(λ::T where {T<:Real},p::T where {T<:Real}) 
    λ > 0 || throw(ArgumentError("correlation length λ of exponential covariance cannot be negative or zero!"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1!"))
    Exponential{promote_type(typeof(λ),typeof(p))}(promote(λ,p)...) 
end

function apply(e::Exponential, x::AbstractMatrix{T}, y::AbstractMatrix{T}) where {T}
    # TODO
end

show(io::IO,e::Exponential) = print(io, "exponential covariance structure with correlation length λ = $(e.λ) and $(e.p)-norm")

## SquaredExponential ##
struct SquaredExponential{T} <: CovarianceStructure
    λ::T
    p::T
end
    
"""
`SquaredExponential(λ, p)`

Create a squared exponential covariance structure with correlation length `λ` and `p`-norm.

Examples:
```
s = SquaredExponential(0.1, 2)
julia> squared exponential covariance structure with correlation length λ = 0.1 and 2.0-norm 

```
"""
function SquaredExponential(λ::T where {T<:Real},p::T where {T<:Real}) 
    λ > 0 || throw(ArgumentError("correlation length λ of exponential covariance cannot be negative or zero!"))
    p >= 1 || throw(ArgumentError("in p-norm, p must be greater than or equal to 1!"))
    SquaredExponential{promote_type(typeof(λ),typeof(p))}(promote(λ,p)...) 
end

function apply(s::SquaredExponential, x::AbstractMatrix{T}, y::AbstractMatrix{T}) where {T}
    # TODO
end

show(io::IO,s::SquaredExponential) = print(io, "squared exponential covariance structure with correlation length λ = $(s.λ) and $(s.p)-norm")
