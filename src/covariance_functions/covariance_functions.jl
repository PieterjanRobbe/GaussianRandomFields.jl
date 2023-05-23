## covariance_functions.jl : utilities for Gaussian random field covariance functions

## CovarianceStructure ##
"""
Abstract type `CovarianceStructure`

# Examples
```jldoctest
julia> Exponential{Float64} <: CovarianceStructure{Float64}
true

```
See also: [`IsotropicCovarianceStructure`](@ref), [`AnisotropicCovarianceStructure`](@ref)
"""
abstract type CovarianceStructure{T<:Real} end

"""
Abstract type `IsotropicCovarianceStructure <: CovarianceStructure`

# Examples
```jldoctest
julia> Exponential{Float64} <: IsotropicCovarianceStructure{Float64}
true

julia> AnisotropicExponential{Float64} <: IsotropicCovarianceStructure{Float64}
false

```
See also: [`Exponential`](@ref), [`Linear`](@ref), [`Spherical`](@ref), [`Whittle`](@ref), [`Gaussian`](@ref), [`SquaredExponential`](@ref), [`Matern`](@ref)
"""
abstract type IsotropicCovarianceStructure{T} <: CovarianceStructure{T} end

"""
Abstract type `AnisotropicCovarianceStructure <: CovarianceStructure`

# Examples
```jldoctest
julia> AnisotropicExponential{Float64} <: AnisotropicCovarianceStructure{Float64}
true

julia> Exponential{Float64} <: AnisotropicCovarianceStructure{Float64}
false

```
See also: [`AnisotropicExponential`](@ref)
"""
abstract type AnisotropicCovarianceStructure{T} <: CovarianceStructure{T} end

## CovarianceFunction ##
"""
Abstract type `AbstractCovariancFunction`

See also: [`CovarianceFunction`](@ref), [`SeparableCovarianceFunction`](@ref)
"""
abstract type AbstractCovarianceFunction{d} end

# return number of dimension
Base.ndims(::AbstractCovarianceFunction{d}) where d = d

struct CovarianceFunction{d, C<:CovarianceStructure} <: AbstractCovarianceFunction{d}
    cov::C

    function CovarianceFunction{d, C}(cov::C) where {d, C}
        d > 0 || throw(DomainError(d, "dimension must be positive, got $(d)"))
        new{d,C}(cov)
    end
end

"""
    CovarianceFunction(d, cov)

Covariance function in `d` dimensions with covariance structure `cov`.

# Examples
```jldoctest
julia> CovarianceFunction(1, Exponential(0.1))
1d exponential covariance function (λ=0.1, σ=1.0, p=2.0)

julia> CovarianceFunction(2, Matern(0.1, 1.0))
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

```
"""
CovarianceFunction(d::Integer, cov::CovarianceStructure) =
    CovarianceFunction{d,typeof(cov)}(cov)

# return standard deviation of the Gaussian random field
Statistics.std(cov::AbstractCovarianceFunction) = cov.cov.σ

# evaluate the covariance function
"""
    apply(cov, pts...)

Returns the covariance matrix, i.e., the covariance function `cov` evaluated in the points `x`.

# Examples
```jldoctest
julia> exponential_covariance = CovarianceFunction(1, Exponential(1))
1d exponential covariance function (λ=1.0, σ=1.0, p=2.0)

julia> pts = range(0, stop=1, length=11)
0.0:0.1:1.0

julia> C = apply(exponential_covariance, pts);

julia> heatmap(C);

julia> whittle_covariance = CovarianceFunction(2, Whittle(1))
2d Whittle covariance function (λ=1.0, σ=1.0, p=2.0)

julia> C = apply(whittle_covariance, pts, pts);

julia> heatmap(C);

```
See also: [`CovarianceFunction`](@ref), [`Exponential`](@ref), [`Whittle`](@ref)
"""
apply(cov::AbstractCovarianceFunction{d}, x::Vararg{Any, d}) where d = apply(cov.cov, x, x)

# element type of covariance
Base.eltype(::CovarianceStructure{T}) where T = T
Base.eltype(::CovarianceFunction{d,<:CovarianceStructure{T}}) where {d,T} = T

# apply for isotropic random fields
apply(cov::IsotropicCovarianceStructure, dx::Vector{<:Real}) = apply(cov, norm(dx, cov.p))

# evaluate when pts is given as a kron product of 1d points
function apply(cov::CovarianceStructure, x::NTuple{d,AbstractVector},
               y::NTuple{d,AbstractVector}) where d
    C = zeros(eltype(cov), prod(length, x), prod(length, y))
    if  size(C, 1) == size(C, 2)
        return apply_symmetric!(C, cov, x, y)
    else
        return apply_non_symmetric!(C, cov, x, y)
    end
end

function apply_symmetric!(C::Matrix, cov::CovarianceStructure, x::NTuple{d,AbstractVector},
                          y::NTuple{d,AbstractVector}) where d
    z = Vector{eltype(C)}(undef, d)
    xiterator = enumerate(Iterators.product(x...))
    for (j, idy) in enumerate(Iterators.product(y...))
        for (i, idx) in Iterators.take(xiterator, j)
            @. z = idx - idy
	        @inbounds C[i, j] = apply(cov, z)
        end
    end
    Symmetric(C, :U)
end

function apply_non_symmetric!(C::Matrix, cov::CovarianceStructure,
                              x::NTuple{d,AbstractVector},
                              y::NTuple{d,AbstractVector}) where d
    z = Vector{eltype(C)}(undef, d)
    xiterator = enumerate(Iterators.product(x...))
    for (j, idy) in enumerate(Iterators.product(y...))
        for (i, idx) in xiterator
            @. z = idx - idy
            @inbounds C[i, j] = apply(cov, z)
        end
    end
    C
end

# evaluate when pts is given as a Finite Element mesh
function apply(cov::CovarianceStructure, tx::NTuple{2,AbstractMatrix},
               ty::NTuple{2,AbstractMatrix})
    x, y = first(tx), first(ty) # select FE nodes
    size(x) == size(y) || throw(DimensionMismatch())
    d, n = size(x)

    C = zeros(eltype(cov), n, n)
    z = Vector{eltype(C)}(undef, d)
    @inbounds for j in 1:n
        yj = view(y, :, j)
        for i in 1:j
            z .= view(x, :, i) .- yj
            C[i, j] = apply(cov, z)
        end
    end
    Symmetric(C, :U)
end

# evaluate for unstructured grids
apply(cov::CovarianceFunction{d}, x::AbstractMatrix{T}, y::AbstractMatrix{N}) where {d, T, N} = apply(cov.cov, (x,x), (x,x))

# evaluate for KL eigenfunctions
function apply(cov::CovarianceStructure, tx::NTuple{2,AbstractMatrix}, y::Tuple)
    x = first(tx) # select FE nodes
    d, nx = size(x)
    d == size(y, 1) || throw(DimensionMismatch())

    C = zeros(eltype(cov), nx, prod(length, y))
    z = Vector{eltype(C)}(undef, d)
    for (j, idy) in enumerate(Iterators.product(y...))
        for i in 1:nx
            z .= view(x, :, i) .- idy
            @inbounds C[i, j] = apply(cov, z)
        end
    end
    C
end

function Base.show(io::IO, cov::CovarianceFunction)
    print(io, ndims(cov), "d ", shortname(cov.cov), " covariance function")
    showparams(io, cov.cov)
end

function Base.show(io::IO, cov::CovarianceStructure)
    print(io, shortname(cov))
    showparams(io, cov)
end

function showparams(io::IO, cov::CovarianceStructure)
    n = nfields(cov)
    if n > 0
        names = fieldnames(typeof(cov))
        firstname = names[1]
        print(io, " (", firstname, "=", getfield(cov, firstname))
        for i in 2:n
            name = names[i]
            print(io, ", ", name, "=", getfield(cov, name))
        end
        print(io, ")")
    end
end
