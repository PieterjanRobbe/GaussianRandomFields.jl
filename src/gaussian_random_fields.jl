# TODO doctest
# TODO FEM, split up in different file
## gaussian_random_fields.jl ##




abstract type GaussianRandomFieldGenerator end


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

function GaussianRandomField(mean::Vector{T},cov::CovarianceFunction{d,C} where {C},method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{N};mode="nodes",kwargs...) where {d,T<:Real,N<:Int}
    size(p,2) == d || throw(DimensionMismatch("second dimension of points must be equal to $(d)"))
    size(t,2) == d+1 || throw(DimensionMismatch("second dimension of nodes must be equal to $(d+1)"))
    if mode == "center"
        length(mean) == size(t,1) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
        pts = compute_centers(p,t)'
        tri = Matrix{N}(0,0)
    elseif mode == "nodes"
        length(mean) == size(p,1) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
        pts = p'
        tri = t'
    else
        throw(ArgumentError("unknown mode $(mode)"))
    end
    _GaussianRandomField(mean,cov,method,pts,tri;kwargs...)
end

GaussianRandomField(cov::CovarianceFunction,method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{N} where {N<:Int};kwargs...) where {T<:Real} = GaussianRandomField(0,cov,method,p,t;kwargs...)

function GaussianRandomField(mean::N where {N<:Real},cov::CovarianceFunction,method::GaussianRandomFieldGenerator,p::Matrix{T},t::Matrix{N} where {N<:Int};mode="nodes",kwargs...) where {T<:Real}
    if mode == "center"
        M = mean*ones(T,size(t,1))
    elseif mode == "nodes"
        M = mean*ones(T,size(p,1))
    else
        throw(ArgumentError("unknown mode $(mode)"))
    end
    GaussianRandomField(M,cov,method,p,t;mode=mode,kwargs...)
end

function compute_centers(p,t)
    n = size(t,1)
    d = size(p,2)
    pts = zeros(n,d) 	 
    for i in 1:d
        x = p[t[:],i]
        x = reshape(x,size(t))
        pts[:,i] = mean(x,2)
    end
    pts
end

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
    # TODO : do reshape here; we only need one _sample function
end

shape(grf::GaussianRandomField) = length.(grf.pts)
shape(grf::GaussianRandomField{C,M,Tuple{T1,T2}}) where {C,M,T1<:AbstractMatrix,T2<:AbstractMatrix} = size(grf.pts[1],2)

function show(io::IO,grf::GaussianRandomField{C,M,P}) where {C,M,P}
    str =  string(length.(grf.pts))
    str = join(split(str[2:end-1],", "),"x")
    print(io, "Gaussian random field with $(grf.cov) on a $(str) structured grid, using a $(M())")
end

show(io::IO,grf::GaussianRandomField{C,M,Tuple{T1,T2}}) where {C,M,T1<:AbstractMatrix,T2<:AbstractMatrix} = print(io, "Gaussian random field with $(grf.cov) on a mesh with $(size(grf.pts[1],2)) points and $(size(grf.pts[2],2)) elements, using a $(M())")
