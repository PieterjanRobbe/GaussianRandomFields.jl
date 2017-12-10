## gaussian_random_fields.jl ##

"""
`GaussianRandomField{T,N}`

Implements a Gaussian random field.
"""
mutable struct GaussianRandomField{C,T}
    cov::C
    pts
    data
end

# TODO: is this necessary??
"""
`GaussianRandomField(cov,pts,method)`

Create a Gaussian random field with covariance structure `cov` defined in the points `pts`. The  Gaussian random field sampler `method` van be `Cholesky()` or `KarhunenLoeve()`.

Examples:
```
```
"""
#function GaussianRandomField(cov::CovarianceFunction,pts::AbstractArray{T,N} where {T,N},method::M) where {M<:GaussianRandomFieldGenerator}
#    throw(ArgumentError("No construction method for type $(M) is known"))
#end

# TODO: pts can be:
# -- linspace in 1d case
# -- array of linspaces in nd case OR in separable case
# -- matrix of size N-by-n in nd case
# TODO ignore separable case for now;
# let Cholesky take matrix: linspace; array of lin spaces => check dimensions and apply
#

## TODO takes multiple linspaces ---
# GaussianRandomField(cov::CovarianceFunction{1,T} where {T},pts::StepRangeLen,method::M where {M<:GaussianRandomFieldGenerator}) = GaussianRandomField(cov,collect(pts),method)

function GaussianRandomField(cov::CovarianceFunction{d,T} where {T},method::M where {M<:GaussianRandomFieldGenerator},pts::S...) where {d,S<:StepRangeLen}
    length(pts) == d || throw(ArgumentError("number of point ranges must be equal to dimension of the covariance function!"))
    # collect all points in matrix
    idx = collect(Iterators.flatten(Base.product(pts...)))
    Pts = reshape(idx,(d,prod(length.(pts))))'
    GaussianRandomField(cov,method,Pts)
end

"""
`sample(field::T) where {T<:GaussianRandomField}`

Take a sample of the Gaussian random field `field`.

Example:
```
```
"""
#function sample(field::T where {T<:GaussianRandomFieldGenerator}, xi::Vector{T} where {T<:AbstractFloat})
#    compose(field.sampler,xi)
#end
