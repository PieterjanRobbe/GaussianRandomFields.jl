## covariance_functions.jl : utilities for Gaussian random field covariance functions

## CovarianceStructure ##
abstract type CovarianceStructure{T<:Real} end
abstract type IsotropicCovarianceStructure{T} <: CovarianceStructure{T} end
abstract type AnisotropicCovarianceStructure{T} <: CovarianceStructure{T} end

## CovarianceFunction ##
abstract type AbstractCovarianceFunction{d} end

# return number of dimension
Base.ndims(::AbstractCovarianceFunction{d}) where d = d

struct CovarianceFunction{d,C<:CovarianceStructure} <: AbstractCovarianceFunction{d}
    cov::C

    function CovarianceFunction{d,C}(cov::C) where {d,C}
        d > 0 || throw(DomainError(d, "dimension must be positive, got $(d)"))
        new{d,C}(cov)
    end
end

"""
	CovarianceFunction(d, cov)

Create a covariance function in `d` dimensions with covariance structure `cov`.

# Examples
```jldoctest
julia> m = Matern(0.1,1.0)
Matérn (λ=0.1, ν=1.0, σ=1.0, p=2.0)

julia> c = CovarianceFunction(2,m)
2d Matérn covariance function (λ=0.1, ν=1.0, σ=1.0, p=2.0)

```
"""
CovarianceFunction(d::Int, cov::CovarianceStructure) =
    CovarianceFunction{d,typeof(cov)}(cov)

# return standard deviation of the Gaussian random field
Statistics.std(cov::CovarianceFunction) = cov.cov.σ

# evaluate the covariance function
apply(cov::CovarianceFunction, x, y) = apply(cov.cov, x, y)

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

function show(io::IO, c::CovarianceFunction{d}) where {d}
    str = split(string(c.cov)," ")
    print(io, "$(d)d $(str[1]) covariance function $(join(str[2:end]," "))")
end
