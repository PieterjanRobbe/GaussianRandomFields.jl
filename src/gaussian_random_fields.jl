# TODO doctest
## gaussian_random_fields.jl ##

"""
`GaussianRandomField{C,G}`

Implements a Gaussian random field.
"""
mutable struct GaussianRandomField{C,G,P}
    mean
    cov::C
    pts::P
    data
end

## CovarianceFunction ##
"""
`GaussianRandomField(mean,cov,method,pts)`

Create a Gaussian random field with mean `mean` and covariance structure `cov` defined in the points `pts`. The  Gaussian random field sampler `method` can be `Cholesky()`, `Spectral()` or `KarhunenLoeve(n)` where `n` is the number of terms in the expansion.

Examples:
```
```
"""
function GaussianRandomField(mean::Array{T} where {T<:Real},cov::CovarianceFunction{d,T} where {T},method::M where {M<:GaussianRandomFieldGenerator},pts::V...;kwargs...) where {d,V<:AbstractVector}
    all(size(mean).==length.(pts)) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
    length(pts) == d || throw(DimensionMismatch("number of point ranges must be equal to the dimension of the covariance function"))
    _GaussianRandomField(mean,cov,method,pts...;kwargs...)
end

"""
`GaussianRandomField(cov,method,pts)`

Create a zero-mean Gaussian random field with covariance structure `cov` defined in the points `pts`. The  Gaussian random field sampler `method` can be `Cholesky()`, `Spectral()` or `KarhunenLoeve(n)` where `n` is the number of terms in the expansion.

Examples:
```
```
"""
GaussianRandomField(cov::CovarianceFunction{d,N} where {N<:CovarianceStructure{T}},method::M where {M<:GaussianRandomFieldGenerator},pts::V...;kwargs...) where {d,T,V<:AbstractVector} = GaussianRandomField(zeros(T,length.(pts)...),cov,method,pts...;kwargs...)

GaussianRandomField(mean::Real,cov::CovarianceFunction{d,N} where {N<:CovarianceStructure{T}},method::M where {M<:GaussianRandomFieldGenerator},pts::V...;kwargs...) where {d,T,V<:AbstractVector} = GaussianRandomField(mean*ones(T,length.(pts)...),cov,method,pts...;kwargs...)

function GaussianRandomField(mean::Vector{T},cov::CovarianceFunction{d,N} where {N},method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{T};kwargs...) where {d,T<:Real}
    length(mean) == size(p,1) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
	size(p,2) == d || throw(DimensionMismatch("second dimension of points must be equal to $(d)"))
	size(t,2) == d+1 || throw(DimensionMismatch("second dimension of nodes must be equal to $(d+1)"))
	_GaussianRandomField(mean,cov,method,p',t',kwargs...)
end

GaussianRandomField(cov::CovarianceFunction,method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{T};kwargs...) where {T<:Real} = GaussianRandomField(zeros(T,size(p,1)),cov,method,p,t;kwargs...)
GaussianRandomField(mean::N where {N<:Real},cov::CovarianceFunction,method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{T};kwargs...) where {T<:Real} = GaussianRandomField(mean*ones(T,size(p,1)),cov,method,p,t;kwargs...)

"""
`sample(grf; xi)`

Take a sample from the Gaussian Random Field `grf` using the (optional) random numbers `xi`. The length of `xi` must have appropriate length. The default value is `randn(randdim(grf))`

Examples:
```
```
"""
function sample(grf::GaussianRandomField; xi::Vector{T} where {T<:AbstractFloat} = randn(randdim(grf)) )
    length(xi) == randdim(grf) || throw(DimensionMismatch("length of random points vector must be equal to $(randdim(grf))"))
    _sample(grf,xi)
end
