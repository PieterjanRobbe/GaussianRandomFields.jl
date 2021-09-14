## gaussian_random_fields.jl : Gaussian random field types and generator types

"""
Abstract type `GaussianRandomFieldGenerator`

The following Gaussian random field generators are implemented:
- `Cholesky`: Cholesky factorization of the covariance matrix, exact but expensive for random fields in dimension `d` > 1 
- `Spectral`: spectral (eigenvalue) decomposition of the covariance matrix, exact but expensive for random fields in dimension `d` > 1
- `KarhunenLoeve`: Karhunen-Lo\u00e8ve expansion, inexact but very efficient for "smooth" random fields when used with a low truncation dimension 
- `CirculantEmbedding`: circulant embedding method, exact and efficient, but can only be used for random fields on structured grids

See also: [`Cholesky`](@ref), [`Spectral`](@ref), [`KarhunenLoeve`](@ref), [`CirculantEmbedding`](@ref)
"""
abstract type GaussianRandomFieldGenerator end

"""
    GaussianRandomField([mean,] cov, generator, pts...)
    GaussianRandomField([mean,] cov, generator, nodes, elements)

Compute a Gaussian random field with mean `mean` and covariance structure `cov` defined in the points `pts`, and computed using the Gaussian random field generator `generator`.

# Examples
```jldoctest label2
julia> cov = CovarianceFunction(2, Matern(.3, 1))
2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0)

julia> pts = pts = range(0, stop=1, length=51)
0.0:0.02:1.0

julia> mean = fill(π, (51, 51))
[...]

julia> grf = GaussianRandomField(mean, cov, Cholesky(), pts, pts)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 51×51 structured grid, using a Cholesky decomposition

```
If no `mean` is specified, a zero-mean Gaussian random field is assumed.
```jldoctest label2
julia> grf = GaussianRandomField(cov, Cholesky(), pts, pts)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 51×51 structured grid, using a Cholesky decomposition

```
The  Gaussian random field generator `generator` can be `Cholesky()`, `Spectral()`, `KarhunenLoeve(n)` (where `n` is the number of terms in the expansion), or `CirculantEmbedding()`. The points `pts` can be specified as arguments of type `AbstractVector`, in which case a tensor (Kronecker) product is assumed, or as a Finite Element mesh with node table `nodes` and element table `elements`.
```jldoctest label2
julia> grf = GaussianRandomField(cov, KarhunenLoeve(500), pts, pts)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a 51×51 structured grid, using a KL expansion with 500 terms

julia> exponential_cov = CovarianceFunction(2, Exponential(.1))
2d exponential covariance function (λ=0.1, σ=1.0, p=2.0)

julia> grf = GaussianRandomField(exponential_cov, CirculantEmbedding(), pts, pts)
Gaussian random field with 2d exponential covariance function (λ=0.1, σ=1.0, p=2.0) on a 51×51 structured grid, using a circulant embedding

```
Separable Gaussian random fields can be defined using `SeparableCovarianceFunction`.
```jldoctest label2
julia> separable_cov = SeparableCovarianceFunction(Exponential(.1), Exponential(.1))
2d separable covariance function [ exponential (λ=0.1, σ=1.0, p=2.0), exponential (λ=0.1, σ=1.0, p=2.0) ]

julia> grf = GaussianRandomField(separable_cov, KarhunenLoeve(500), pts, pts)
Gaussian random field with 2d separable covariance function [ exponential (λ=0.1, σ=1.0, p=2.0), exponential (λ=0.1, σ=1.0, p=2.0) ] on a 51×51 structured grid, using a KL expansion with 500 terms

julia> plot_eigenfunction(grf, 3)
[...]

```
We also offer support for anisotropic random fields.
```jldoctest label2
julia> anisotropic_cov = CovarianceFunction(2, AnisotropicExponential([500 400; 400 500]))
2d anisotropic exponential covariance function (A=[500 400; 400 500], σ=1.0)

julia> grf = GaussianRandomField(anisotropic_cov, CirculantEmbedding() , pts, pts)
Gaussian random field with 2d anisotropic exponential covariance function (A=[500 400; 400 500], σ=1.0) on a 51×51 structured grid, using a circulant embedding

julia> heatmap(grf)
[...]

```
For irregular domains, specify the points as matrices containing the `nodes` and `elements` of a finite element mesh. To compute the value of the random field at the element centers, use the optional keyword `mode="center"`.
```jldoctest label2
julia> nodes, elements = Lshape()
[...]

julia> grf = GaussianRandomField(cov, Spectral(), nodes, elements)
Gaussian random field with 2d Matérn covariance function (λ=0.3, ν=1.0, σ=1.0, p=2.0) on a mesh with 998 points and 1861 elements, using a spectral decomposition

```
Alternativly, simply pass a `Matrix{T}` of size `N` by `d` to compute the random field at un unstructured grid defined by the given set of points.

Samples from the random field can be computed using the `sample` function.
```jldoctest label2
julia> sample(grf)
[...]

```
See also: [`Cholesky`](@ref), [`Spectral`](@ref), [`KarhunenLoeve`](@ref), [`CirculantEmbedding`](@ref), [`sample`](@ref)
"""
struct GaussianRandomField{G<:GaussianRandomFieldGenerator,C<:AbstractCovarianceFunction,
                           P,M,D}
    mean::M
    cov::C
    pts::P
    data::D
end

function GaussianRandomField(mean::Array{<:Real}, cov::CovarianceFunction{d}, method::GaussianRandomFieldGenerator, pts::Vararg{AbstractVector,d}; kwargs...) where d
    size(mean) == length.(pts) || throw(DimensionMismatch("size of the mean does not correspond to the dimension of the points"))

    method isa CirculantEmbedding && !(pts isa NTuple{d,AbstractRange}) &&
        throw(ArgumentError("can only use circulant embedding on a regular grid, supply ranges for pts"))

    method isa Union{CirculantEmbedding,KarhunenLoeve} && any(pt -> length(pt) < 2, pts) &&
        throw(ArgumentError("must have at least 2 points in each direction to use circulant embedding or KL expansion"))

    _GaussianRandomField(mean, cov, method, pts...; kwargs...)
end

# zero-mean GRF
GaussianRandomField(cov::CovarianceFunction{d}, method::GaussianRandomFieldGenerator, pts::Vararg{AbstractVector,d}; kwargs...) where d = GaussianRandomField(zeros(eltype(cov), length.(pts)), cov, method, pts...; kwargs...)

# constant mean GRF
GaussianRandomField(mean::Real, cov::CovarianceFunction{d}, method::GaussianRandomFieldGenerator, pts::Vararg{AbstractVector,d}; kwargs...) where d = GaussianRandomField(fill(convert(eltype(cov), mean), length.(pts)), cov, method, pts...; kwargs...)

# obtain generator used to compute random field
generator(::GaussianRandomField{G}) where G = G()

"""
	sample(grf)
        sample(grf[, xi])

Take a sample from the Gaussian random field `grf` using the (optional) normally distributed random numbers `xi`. The vector`xi` must have appropriate length.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Whittle(.1))
2d Whittle covariance function (λ=0.1, σ=1.0, p=2.0)

julia> pts = pts = range(0, stop=1, length=51)
0.0:0.02:1.0

julia> grf = GaussianRandomField(cov, CirculantEmbedding(), pts, pts)
Gaussian random field with 2d Whittle covariance function (λ=0.1, σ=1.0, p=2.0) on a 51×51 structured grid, using a circulant embedding

julia> sample(grf)
[...]

julia> sample(grf, xi=randn(randdim(grf)))
[...]
```
See also: [`GaussianRandomField`](@ref), [`Matern`](@ref), [`CovarianceFunction`](@ref), [`CirculantEmbedding`]
"""
function sample(grf::GaussianRandomField; xi::AbstractArray{<:Real}=randn(randdim(grf)))
    length(xi) == prod(randdim(grf)) || throw(DimensionMismatch("length of random points vector must be equal to $(randdim(grf))"))
    _sample(grf, xi)
end

function Base.show(io::IO, grf::GaussianRandomField)
    print(io, "Gaussian random field with ", grf.cov, " on a")
    showpoints(io, grf.pts)
    print(io, ", using a ", generator(grf))
end

function showpoints(io::IO, points)
    d = length(points)
    if d == 1
        print(io, " ", length(points[1]), "-point")
    else
        print(io, " ", length(points[1]))
        @inbounds for i in 2:d
            print(io, "×", length(points[i]))
        end
    end
    print(io, " structured grid")
end

"""
    randdim(grf)

Returns the number of random numbers used to sample from the Gaussian random field `grf`.

# Examples
```jldoctest
julia> cov = CovarianceFunction(2, Whittle(.1))
2d Whittle covariance function (λ=0.1, σ=1.0, p=2.0)

julia> pts = pts = range(0, stop=1, length=51)
0.0:0.02:1.0

julia> grf = GaussianRandomField(cov, KarhunenLoeve(200), pts, pts)
Gaussian random field with 2d Whittle covariance function (λ=0.1, σ=1.0, p=2.0) on a 51×51 structured grid, using a KL expansion with 200 terms

julia> randdim(grf)
200

```
"""
randdim(grf::GaussianRandomField) = 0
