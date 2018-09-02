## gaussian_random_fields.jl : Gaussian random field types and generator types

abstract type GaussianRandomFieldGenerator end

struct GaussianRandomField{G<:GaussianRandomFieldGenerator,C<:AbstractCovarianceFunction,
                           P,M,D}
    mean::M
    cov::C
    pts::P
    data::D
end

"""
	GaussianRandomField(mean,cov,method,pts...)
	GaussianRandomField(cov,method,pts...)
	GaussianRandomField(mean,cov,method,nodes,elements)
	GaussianRandomField(cov,method,nodes,elements)

Compute a Gaussian random field with mean `mean` and covariance structure `cov` defined in the points `pts`, and computed using the method `method`.

# Examples
```jldoctest
julia> m = Matern(0.1,1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(2,m)
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> pts1 = 0:0.02:1; pts2 = 0:0.02:1 
0.0:0.02:1.0

julia> mn = ones(length(pts1),length(pts2))

julia> grf = GaussianRandomField(mn,c,Cholesky(),pts1,pts2)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a Cholesky decomposition

```
If no `mean` is specified, a zero-mean Gaussian random field is assumed.
```jldoctest
julia> grf = GaussianRandomField(c,Cholesky(),pts1,pts2)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a Cholesky decomposition

```
The  Gaussian random field sampler `method` can be `Cholesky()`, `Spectral()`, `KarhunenLoeve(n)` (where `n` is the number of terms in the expansion), or `CirculantEmbedding()`. The dimension of the points must match the dimension of the covariance function `cov`. The points can be specified as arguments of type `AbstractVector`, in which case a tensor (Kronecker) product is assumed, or as a Finite Element mesh with node table `nodes` and element table `elements`.
```jldoctest
julia> grf = GaussianRandomField(c,KarhunenLoeve(500),pts1,pts2)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a KL expansion with 500 terms

```
For separable Gaussian random fields with a KL expansion, provide a [`SeparableCovarianceFunction`](@ref). Note that the number of terms `n` in `KarhunenLoeve(n)` refers to the total number of terms.
```jldoctest
julia> e1 = Exponential(0.1); e2 = Exponential(0.01);

julia> scov = SeparableCovarianceFunction(e1,e2)
2d separable covariance function [ exponential (λ=0.1, σ=1.0, p=2.0), exponential (λ=0.01, σ=1.0, p=2.0) ]

julia> grf = GaussianRandomField(scov,KarhunenLoeve(500),pts1,pts2)
Gaussian random field with 2d separable covariance function [ exponential (λ=0.1, σ=1.0, p=2.0), exponential (λ=0.01, σ=1.0, p=2.0) ] on a 51x51 structured grid, using a KL expansion with 500 terms

julia> plot_eigenfunction(grf,3); show()
[...]

```
Also anisotropic Gaussian random fields can be computed. For example, the anisotropic exponential covariance function needs a positive definite matrix `A`. The size of the off-diagonal elements of `A` determine the degree of anisotropy.
```jldoctest
julia> a = AnisotropicExponential([1 0.8;0.8 1])
anisotropic exponential (A=[1.0 0.8; 0.8 1.0], σ=1.0)

julia> acov = CovarianceFunction(2,a)
2d anisotropic covariance function exponential (A=[1.0 0.8; 0.8 1.0], σ=1.0)

julia> pts = range(0,stop = 10,length = 128)
0.0:0.07874015748031496:10.0

julia> grf = GaussianRandomField(acov,CirculantEmbedding(),pts,pts)
WARNING: negative eigenvalue -1.2245106889248049e-18 detected, Gaussian random field will be approximated (ignoring all negative eigenvalues)
WARNING: increase padding if possible
Gaussian random field with 2d anisotropic covariance function exponential (A=[1.0 0.8; 0.8 1.0], σ=1.0) on a 128x128 structured grid, using a circulant embedding

julia> contourf(grf)
[...]

```
Samples from the random field can be computed using the `sample` function.
```jldoctest
julia> sample(grf)
[...]

```
See also: [`Cholesky`](@ref), [`Spectral`](@ref), [`KarhunenLoeve`](@ref), [`CirculantEmbedding`](@ref), [`sample`](@ref)
"""
function GaussianRandomField(mean::Array{T} where {T<:Real},cov::CovarianceFunction{d,T} where {T},method::M where {M<:GaussianRandomFieldGenerator},pts::V...;kwargs...) where {d,V<:AbstractVector}
    all(size(mean).==length.(pts)) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))
    length(pts) == d || throw(DimensionMismatch("number of point ranges must be equal to the dimension of the covariance function"))
	( typeof(method) <: CirculantEmbedding && !(V<:AbstractRange) ) && throw(ArgumentError("can only use circulant embedding on a regular grid, supply ranges for pts"))
	( ( typeof(method) <: CirculantEmbedding || typeof(method) <: KarhunenLoeve ) && any(length.(pts).<2) ) && throw(ArgumentError("must have at least 2 points in each direction to use circulant embedding or KL expansion"))
    _GaussianRandomField(mean,cov,method,pts...;kwargs...)
end

# zero-mean GRF
GaussianRandomField(cov::CovarianceFunction{d,N} where {N<:CovarianceStructure{T}},method::M where {M<:GaussianRandomFieldGenerator},pts::V...;kwargs...) where {d,T,V<:AbstractVector} = GaussianRandomField(zeros(T,length.(pts)...),cov,method,pts...;kwargs...)

# constant mean GRF
GaussianRandomField(mean::Real,cov::CovarianceFunction{d,N} where {N<:CovarianceStructure{T}},method::M where {M<:GaussianRandomFieldGenerator},pts::V...;kwargs...) where {d,T,V<:AbstractVector} = GaussianRandomField(mean*ones(T,length.(pts)...),cov,method,pts...;kwargs...)
# obtain generator used to compute random field
generator(::GaussianRandomField{G}) where G = G()

"""
	sample(grf)
	sample(grf, xi=randn(randdim(grf)))

Take a sample from the Gaussian Random Field `grf` using the (optional) random numbers `xi`. The vector`xi` must have appropriate length. The default value is `randn(randdim(grf))`.

# Examples
```
julia> m = Matern(0.1,1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(2,m)
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> pts1 = 0:0.02:1; pts2 = 0:0.02:1 
0.0:0.02:1.0

julia> grf = GaussianRandomField(c,KarhunenLoeve(300),pts1,pts2)
Gaussian random field with 2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0) on a 51x51 structured grid, using a KL expansion with 300 terms

julia> sample(grf)
[...]

julia> sample(grf,xi=2*rand(randdim(grf))-1)
[...]

```
"""
function sample(grf::GaussianRandomField; xi::AbstractArray{<:Real}=randn(randdim(grf)))
    length(xi) == prod(randdim(grf)) || throw(DimensionMismatch("length of random points vector must be equal to $(randdim(grf))"))
    _sample(grf,xi)
end

function Base.show(io::IO, grf::GaussianRandomField)
    print(io, "Gaussian random field with ", grf.cov, " on a ")
    showpoints(io, grf.pts)
    print(io, ", using a ", generator(grf))
end

function showpoints(io::IO, points)
    d = length(points)
    if d == 1
        print(io, length(points[1]), "-point")
    else
        print(io, length(points[1]))
        @inbounds for i in 2:d
            print(io, "×", length(points[i]))
        end
    end
    print(io, " structured grid")
end
